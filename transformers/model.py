# import basic torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval= 300
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embed = 32
n_heads = 4

torch.manual_seed(42)

with open('shakespeare.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda x : [stoi[c] for c in x]
decode = lambda x : [itos[c] for c in x]

data = torch.tensor(encode(text),dtype=torch.long)

split = int(len(data) * 0.9)
train_data = data[:split]
test_data = data[split:]


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits,loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    def forward(self,x):
        # x : (B,T,n_embed)
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x)
        w = q @ k.transpose(-1,-2) / (C ** 0.5) # (B,T,T)
        w = w.masked_fill(self.tril == 0, float('-inf'))
        w = F.softmax(w,dim=-1)
        v = self.value(x)
        out = w @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size)] for _ in range(num_heads))
    
    def forward(self,X):
        return torch.cat([head(X) for head in self.heads],dim=-1)
    

class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,n_embed)
        self.positon_embedding = nn.Embedding(block_size,n_embed)
        self.sa_head = MultiHeadAttention(n_heads,n_embed // n_heads)
        self.lm_head = nn.Linear(n_embed,vocab_size)
    
    def forward(self,x,targets=None):
        # x : (B,T)
        B,T = x.shape 
        tok_emb = self.embedding_table(x) # B,T,n_embed
        pos_emb = self.positon_embedding(torch.arange(T,device=device)) # T,n_embed
        x = tok_emb + pos_emb # B,T,n_embed
        x = self.sa_head(x)
        logits = self.lm_head(x) # B,T,vocab_size
        if targets!=None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        else:
            loss = None
        return logits,loss

model = TransformerDecoder()
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    x,y = get_batch("train")
    logits,loss = model(x,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0:
        x,y = get_batch("test")
        logits,loss = model(x,y)
        print(f"iter {iter} : loss = {loss.item():.4f}")


