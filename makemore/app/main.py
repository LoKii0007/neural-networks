from pathlib import Path
import torch
from save_image import save_matrix_image

names_path = Path(__file__).with_name("names.txt")
words = names_path.read_text(encoding="utf-8").splitlines()

print("lenght -", len(words))

b = {}  # to keep count
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        biagram = (ch1, ch2)
        b[biagram] = b.get(biagram, 0) + 1

sorted_items = sorted(b.items(), key=lambda kv: -kv[1])

# print(sorted_items)

# instead of dictionary we need to set these in an array

chars = sorted(list(set("".join(words))))

# print('', chars)

N = torch.zeros((27, 27), dtype=torch.int32)

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}
# print(itos)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print(N[3, 3].item())

# print(N)

# saved_path = save_matrix_image(N, itos=itos)
# print('saved image -', saved_path)

# we need to convert the number into probablities to get a better understanding

# p = N[0].float()
# p = p / p.sum()
# print(p)

g = torch.Generator().manual_seed(2147483647)
# ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g.item())
# ix = itos[ix]
# print(ix)

print("-----------------")

# P = [(N[i].float()) / (N[i].float()).sum() for i in range(27)]
# print(P)

P = N.float()
P =P/ P.sum(1, keepdim = True)


for i in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))
