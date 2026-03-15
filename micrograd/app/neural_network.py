from main import Value
import random
from viz import save_graph


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]
        
    
x = [2.0, 3.0]
n = Neuron(2)
# n(x)

l = Layer(2, 3)
# l(x)

m = MLP(2, [4, 4, 1])
print(m(x))
# params = m.parameters()
# print(len(params))
# save_graph(m(x))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
m = MLP(3, [4,4,1])

# print(yc)

## manually calculating loss to fi thw eights so that nn performs well 
# yc = [m(x) for x in xs]
# params = m.parameters()
# loss = sum([ (yc- ye)**2 for yc, ye in zip(yc, ys) ])

# print('--------')
# print('before adjusting params')

# loss.backward()
# # save_graph(loss)
# print('grad', m.layers[0].neurons[0].w[0].grad)
# print('data', m.layers[0].neurons[0].w[0].data)
# print('loss', loss)

# for p in params:
#     p.data += 0.01 * p.grad

# yc = [m(x) for x in xs]
# loss = sum([ (yc- ye)**2 for yc, ye in zip(yc, ys) ])
# print('--------')
# print('after adjusting params')

# print('grad', m.layers[0].neurons[0].w[0].grad)
# print('data', m.layers[0].neurons[0].w[0].data)


# print('loss',loss)

# creating a loop


for k in range(20):
    
    # forward pass 
    yc = [m(x) for x in xs]
    loss = sum([ (yc- ye)**2 for yc, ye in zip(yc, ys) ])
    
    # backward pass 
    for p in m.parameters(): #for each backward pass we need to reset the grads to 0 otherwise its += so they are being updated with their prev values
        p.grad = 0.0
    loss.backward()
    
    # update 
    for p in m.parameters():
        p.data += 0.05 * p.grad
        
    # values
    print('step - ', k+1, 'loss - ', loss) 
    print(yc)