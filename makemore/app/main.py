from pathlib import Path
import torch
from save_image import save_matrix_image

names_path = Path(__file__).with_name('names.txt')
words = names_path.read_text(encoding='utf-8').splitlines()

print('lenght -', len(words))

b = {} #to keep count 
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        biagram = (ch1, ch2)
        b[biagram] = b.get(biagram, 0) + 1
        
sorted_items = sorted(b.items(), key = lambda kv:-kv[1])

# print(sorted_items)

# instead of dictionary we need to set these in an array

chars = sorted(list(set(''.join(words))))

print('', chars)

N = torch.zeros((28,28), dtype = torch.int32)

stoi = {s:i for i, s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

itos = {i:s for s,i in stoi.items()}
print(itos)

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
        
print(N) 

saved_path = save_matrix_image(N, itos=itos)
print('saved image -', saved_path)