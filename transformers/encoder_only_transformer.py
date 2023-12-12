import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

n_embed = 40
head_size = 10
n_heads = 4
vocab_size = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 32
n_layers = 4

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
    def forward(self,x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        w = k @ q.transpose(-2,-1)
        w = F.softmax(w,dim=-1)
        out = w @ v
        return out

class MultiHead(nn.Module):
    def __init__(self,head_size,n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads],-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,n_embed),
            nn.ReLU(),
            nn.Linear(n_embed,n_embed),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead = MultiHead(head_size,n_heads)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self,x):
        x = self.ln1(x)
        x = x + self.multihead(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,n_embed)
        self.positional_embedding = nn.Embedding(block_size,n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)
        self.cl_head = nn.Sequential(
            nn.Linear(n_embed,2),
        )
    def forward(self,x,targets=None):
        ini_emb = self.embedding(x)
        pos_emb = self.positional_embedding(torch.arange(block_size,device=device))
        x = ini_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        x = self.cl_head(x)
        if targets != None:
            loss = F.cross_entropy(x,targets)
            return x,loss
        else:
            return x
    