# import basic torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval= 300
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200

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

class BigramModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx,targets=None):
        logits = self.embedding_table(idx)
        if targets!=None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        else:
            loss = None
        return logits,loss

model = BigramModel(vocab_size)
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


