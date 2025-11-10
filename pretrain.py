import math
import time
import torch
import json
import os
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from nanoGPT import GPTConfig, GPT, draw_losses, draw_val_losses, eval

class WikiDataset(Dataset):
    def __init__(self, config, tokenizer, folder, max_texts=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = config.block_size
        self.eos_token_id = tokenizer.eos_token_id
        self.encoded_chunks = []

        folder = Path(folder)
        files = sorted([p for p in folder.iterdir() if p.name.startswith("wiki_")])
        count = 0
        buffer_tokens = []

        for p in files:
            with p.open('r', encoding='utf8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    text = j.get("text", "").strip()
                    if not text:
                        continue

                    ids = tokenizer.encode(text)
                    buffer_tokens.extend(ids)

                    while len(buffer_tokens) >= (self.block_size + 1):
                        chunk = buffer_tokens[:self.block_size + 1]
                        self.encoded_chunks.append(chunk)
                        buffer_tokens = buffer_tokens[self.block_size:]
                    count += 1
                    if max_texts is not None and count >= max_texts:
                        break
            if max_texts is not None and count >= max_texts:
                break

        print(f'>>> WikiDataset initialized: folder={folder.name}, files={len(files)}, texts_used={count}, chunks={len(self.encoded_chunks)}')

    def __len__(self):
        return len(self.encoded_chunks)

    def __getitem__(self, idx):
        chunk = self.encoded_chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class MixedDataset(Dataset):
    def __init__(self, config, tokenizer, folders, max_texts=None):
        super().__init__()
        self.encoded_chunks = []
        for folder in folders:
            d = WikiDataset(config, tokenizer, folder, max_texts=max_texts)
            self.encoded_chunks.extend(d.encoded_chunks)
        random.shuffle(self.encoded_chunks)  # global shuffle（像把所有题洗牌打散）
        print(f">>> MixedDataset initialized: total_chunks={len(self.encoded_chunks)}")

    def __len__(self):
        return len(self.encoded_chunks)

    def __getitem__(self, idx):
        chunk = self.encoded_chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_lr_scheduler(it, total_steps, warmup_steps, max_lr, min_lr):
    """根据当前迭代步数计算学习率 (Warmup + Cosine Decay)"""
    # 1. Warmup phase (0 -> max_lr)
    if it < warmup_steps:
        return max_lr * (it / warmup_steps)
    # 2. Cosine Decay phase (max_lr -> min_lr)
    if it > total_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine formula
    return min_lr + coeff * (max_lr - min_lr)

def save_checkpoint(model, optimizer, tag, run_folder, val_loss):
    path = Path(run_folder) / f'{tag}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)
    print(f">>> Checkpoint saved: {path}")


def train(device, model, optimizer, scaler, train_loader, val_loader,
          run_folder, max_epochs, accumulation_steps):

    run_folder = Path(run_folder)
    val_losses = []

    total_steps = len(train_loader) * max_epochs
    global_step = 0
    warmup_steps = int(0.1 * total_steps)  # 10% steps for warmup
    max_lr = optimizer.param_groups[0]['lr']
    min_lr = max_lr / 10  # 设置最小 LR 为最大 LR 的 1/10

    for epoch in range(1, max_epochs+1):

        model.train()
        losses = []

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            lr = get_lr_scheduler(global_step, total_steps, warmup_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type='cuda' if device.type.startswith('cuda') else 'cpu'):
                logits, loss = model(x, targets=y)
                losses.append(loss.item())
                loss=loss/accumulation_steps
            scaler.scale(loss).backward()

            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f'>>> Epoch {epoch}, Batch: {batch_idx}, Loss: {losses[-1]:.4f}')

            global_step += 1

        draw_losses(run_folder, epoch, losses, True)
        val_loss = eval(model, val_loader, device)
        val_losses.append(val_loss)
        save_checkpoint(model, optimizer, f'epoch{epoch}', run_folder, val_loss)

    draw_val_losses(run_folder, val_losses, True)

def main():
    max_epochs = 20
    accumulation_steps = 32
    max_texts = 10000

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder = f"runs/run_{timestamp}"
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    config = GPTConfig()
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)

    # resume_path = "runs/run_20251108_152338/epoch9.pt"
    # if os.path.exists(resume_path):
    #     print(">>> Loading checkpoint:", resume_path)
    # checkpoint = torch.load(resume_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    folders = [
        "wiki_zh_2019/wiki_zh/AA",
        "wiki_zh_2019/wiki_zh/AB",
        "wiki_zh_2019/wiki_zh/AC"
    ]

    dataset = MixedDataset(config, tokenizer, folders, max_texts=max_texts)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    train(
        device=device,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        run_folder=run_folder,
        max_epochs=max_epochs,
        accumulation_steps=accumulation_steps
    )

    print(">>> Pretraining finished.")

if __name__ == "__main__":
    main()