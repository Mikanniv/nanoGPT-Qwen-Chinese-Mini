import re
import os
import time
from pathlib import Path
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from nanoGPT import GPT, GPTConfig, draw_losses, draw_val_losses, eval

class SFTDataset(Dataset):
    def __init__(self, config, data_split):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2-0.5B-Instruct', trust_remote_code=True
        )

        special_tokens = {'special_tokens': ['<human>', '<assistant>']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.eos_token_id = self.tokenizer.eos_token_id

        # 加载原始数据
        dataset = load_dataset('BelleGroup/multiturn_chat_0.8M', split=data_split)
        print("Sample from dataset:", dataset[0])

        raw_data = []
        for d in dataset:
            instr = d.get('instruction', '')
            single_turns = self.split_single_turn(instr)
            for human, assistant in single_turns:
                text = f"<human> {human} <assistant> {assistant}"
                text = self.clean_text(text)
                if text:
                    raw_data.append(text)

        # 编码成连续 token
        full_encoded = []
        for text in raw_data:
            tokens = self.tokenizer.encode(text)
            full_encoded.extend(tokens)

        # 按 block_size 切片
        self.encoded_data = []
        for i in range(0, len(full_encoded), config.block_size + 1):
            chunk = full_encoded[i:i + config.block_size + 1]
            if len(chunk) < config.block_size + 1:
                chunk += [self.eos_token_id] * (config.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

        print(f'total samples: {len(self.encoded_data)}')

    @staticmethod
    def clean_text(text):
        # 去掉多余空格和特殊字符
        text = text.replace('<|endoftext|>', '')
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9，。！？,.!?：:；;()\[\]\"\'\s]", "", text)
        return text

    @staticmethod
    def split_single_turn(text):
        """
        将多轮 Human/Assistant 对话拆分为单轮 (human, assistant)
        """
        pattern = re.compile(r'(Human\s*[:：]|Assistant\s*[:：])')
        turns = pattern.split(text)
        turns = [t.strip() for t in turns if t.strip()]
        single_turns = []

        i = 0
        while i < len(turns) - 1:
            label = turns[i]
            content = turns[i + 1]
            if 'Human' in label:
                human_text = content
                # 找下一个 Assistant
                if i + 2 < len(turns) and 'Assistant' in turns[i + 2]:
                    assistant_text = turns[i + 3] if i + 3 < len(turns) else ''
                    if human_text and assistant_text:
                        single_turns.append((human_text, assistant_text))
                    i += 4
                else:
                    i += 2
            else:
                i += 2
        return single_turns

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train(device, model, optimizer, scheduler, scaler,
          train_loader, val_loader, run_folder, max_epochs, accumulation_steps):

    run_folder = Path(run_folder)
    val_losses = []

    for epoch in range(1, max_epochs+1):

        model.train()
        losses = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type='cuda' if device.type.startswith('cuda') else 'cpu'):
                logits, loss = model(x, targets=y)
                losses.append(loss.item())
                loss = loss/accumulation_steps
            scaler.scale(loss).backward()

            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f'>>> Epoch {epoch}, Batch: {batch_idx}, Loss: {losses[-1]:.4f}')

        scheduler.step()
        val_loss = eval(model, val_loader, device)
        val_losses.append(val_loss)
        draw_losses(run_folder, epoch, losses, False)

    draw_val_losses(run_folder, val_losses, False)

def main():
    max_epochs = 20
    accumulation_steps = 16

    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = f"runs/run_{timestamp}"
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    config = GPTConfig()
    model = GPT(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    resume_path = ''
    if os.path.exists(resume_path):
        print(">>> Loading checkpoint:", resume_path)
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    dataset = SFTDataset(config, 'train[:30%]')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    train(
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        run_folder=run_folder,
        max_epochs=max_epochs,
        accumulation_steps=accumulation_steps
    )

    print('>>> SFT finished.')

if __name__ == '__main__':
    main()
