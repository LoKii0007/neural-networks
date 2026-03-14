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
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    
x = [2.0, 3.0]
n = Neuron(2)
# n(x)

l = Layer(2, 3)
# l(x)

m = MLP(2, [4, 4, 1])
# print(m(x))
# save_graph(m(x))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
m = MLP(3, [4,4,1])
yc = [m(x) for x in xs]
print(yc)

# calculating loss to fi thw eights so that nn performs well 

loss = sum([ (yc- ye)**2 for yc, ye in zip(yc, ys) ])
print(loss)

loss.backward()
print(m.layers[0].neurons[0].w[0].grad)