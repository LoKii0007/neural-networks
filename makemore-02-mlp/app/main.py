from pathlib import Path
import torch
import torch.nn.functional as F

names_path = Path(__file__).with_name("names.txt")
words = names_path.read_text(encoding="utf-8").splitlines()

print("length -", len(words))

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

print("-----------------")
print("building the dataset")
print("-----------------")

block_size = 3
X , Y = [] , []
for w in words[:5]:
    print('-----')
    print(w)
    print('-----')
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        
        print(''.join(itos[i] for i in context ), '--->', ch)
        context = context[1:] + [ix]
        
X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.rand((27, 2))

