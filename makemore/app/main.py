from pathlib import Path
import torch
from save_image import save_matrix_image, save_matrix_image_default
import torch.nn.functional as F

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

#? instead of dictionary we need to set biagram in an array

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

#? we need to convert the number into probablities to get a better understanding

# p = N[0].float()
# p = p / p.sum()
# print(p)

g = torch.Generator().manual_seed(2147483647)
# ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g.item())
# ix = itos[ix]
# print(ix)

print("-----------------")
print("working name genrator using just stats/prob")
print("-----------------")

# P = [(N[i].float()) / (N[i].float()).sum() for i in range(27)]
# print(P)

model_smoothing = 1
P = (N+model_smoothing).float()
P =P/ P.sum(1, keepdim = True)


for i in range(0):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print("".join(out))


#? how can we make this model better? 
#>  by maximizing the likelihood of the data w.r.t model parameters
    #> to get better results
    #> maximize log likelihood
    #> minimize negative log likelihood 
    
print("-----------------")
print("how to improve?")
print("-----------------")

n = 0
log_likelihood = 0.0
for w in ["lokeshzf"]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1
        # print(f' {ch1} {ch2} , {prob: .4f}, {log_prob: .4f}')
        
# print(f'{ log_likelihood= }')
nll = -log_likelihood
# print(f'nll = { nll }')
print(f'avg nll = { nll/n }')

#> if the probablity of any following char is 0 then we will get inf in log we use some model_smoothing to prevent that

print("-----------------")
print("training")
print("-----------------")
#? we need to train the data

xs = []
ys = []

#? what are xs and ys
    #> exmaple xs = [0,5,13,13,1] -> . e m m a
    #> exmaple ys = [5,13,13,1,.] -> e m m a .
    
    #> we are saying given this car predict the next one
    #* xs is input , ys is the label (what output do we want)
    
    
for w in words[:1]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs, num_classes = 27).float()
# save_matrix_image_default(xenc)
# print(xenc)

#? what is xenc why did we converted xs, ys into that?
    #> xs and ys are just integers and we cant feed raw numbers to neural networks . WHY?
    #> becuse we dont want our neural network to have any meaning about the integers like say 13 is more than 5 . we dont want our neural network to understand that 13 > 5 in any meaningfull way. They are JUST IDS
    
    #> one_hot encode converts the integer into a vector of 27 zeroes with a single 1 at the index
    #> so xenc -> is [5,27] array -> 5 training exmaples , each represented in a 27 dim vector


w = torch.randn((27,27))
logits = xenc @ w  #* log-counts
counts = logits.exp() #* equivalent to N
probs = counts /counts.sum(1, keepdims = True)
# print(probs)

#? why did we multiply the xenc with w ???
    #> w -> weight matrix -> actual neural network -> one layer, 27 inputs, 27 outputs
    #> foe each input character , we get 27 ouput numbers -> one for each possible next character -> THESE ARE CALLED LOGITS (raw scores - not prob yet)
    #> since we are multiplying the w with one_shot we are just selcting the corresponding row
    #> exp to remove -ve values

print("-----------------")
print("working of the neural network")
print("-----------------")

nlls = torch.zeros(5)
for i in range(5):
    x = xs[i].item()
    y = ys[i].item()
    ch1 = itos[x]
    ch2 = itos[y]
    p = probs[i, y]
    log_prob = torch.log(p)
    nll = -log_prob
    nlls[i] = nll
    print(f'the model thinks the prob of {ch2} after {ch1} is {p}')
    print(f' {ch1} {ch2} avg nll =  {nlls[i].item()}')
    print(f'loss = {nlls.mean().item()}')
    print('_____')
        