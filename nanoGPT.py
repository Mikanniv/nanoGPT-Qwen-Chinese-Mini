import math
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
torch.manual_seed(42)

@dataclass
class GPTConfig:
    block_size: int = 256
    batch_size: int = 8
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    hidden_dim: int = 768
    dropout: float = 0.1
    head_size: int = hidden_dim // n_head
    vocab_size: int = 151643

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.head_size = config.head_size
        self.register_buffer('attention_mask', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(self.attention_mask[:seq_len, :seq_len] == 0, float('-inf'))
        weight = F.softmax(weight / math.sqrt(self.head_size), dim=-1)
        weight = self.dropout(weight)
        return weight @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = torch.concat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln = nn.LayerNorm(config.hidden_dim)
    def forward(self, x):
        x = x + self.att(self.ln(x))
        x = x + self.ffn(self.ln(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        batch, seq_len, vocab_size = logits.size()
        logits = logits.view(batch * seq_len, vocab_size)
        targets = targets.view(batch * seq_len)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx (batch_size, seq_len)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            # shape (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # shape (batch_size, seq_len+1)
        return idx

def draw_losses(run_folder, epoch, losses, is_pretrain):
    if is_pretrain:
        path = Path(run_folder) / f"_pretrain_epoch_{epoch}_train_loss.png"
    else:
        path = Path(run_folder) / f"_sft_epoch_{epoch}_train_loss.png"
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title(f"Train Loss Curve(Epoch {epoch})")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f">>> Epoch loss plotted: {path}")

def draw_val_losses(run_folder, val_losses, is_pretrain):
    if is_pretrain:
        path = Path(run_folder) / f"_pretrain_val_loss.png"
    else:
        path = Path(run_folder) / f"_sft_val_loss.png"

    plot = plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title(f"Val Loss Curve")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f">>> Val loss plotted: {plot}")

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (x, y) in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss/len(val_loader)