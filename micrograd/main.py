from viz import save_graph

class Value:
    
    def __init__(self, data, _chidren = (), _op = "", label="", grad = 0.0):
        self.data = data
        self._prev = set(_chidren)
        self._op = _op
        self.label = label
        self.grad = grad
    
    def __repr__(self):
        return f"Value(data={self.data})"
        
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out
    
    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), "-")
        return out


def lol():
    
    h = 0.001
    
    a = Value(data=4.0, label = "a")
    b = Value(data=2.0, label = "b")
    c = Value(data=1.0, label = "c")
    d = a + b
    d.label = "d"
    out = d * c
    out.label = "out"
    out.grad = 1.0
    L1 = out.data
    print(L1)
    
    
    a = Value(data=4.0 , label = "a")
    b = Value(data=2.0, label = "b")
    c = Value(data=1.0, label = "c")
    d = a + b
    d.label = "d"
    out = d * c
    out.label = "out"
    out.grad = 1.0
    L2 = out.data + h
    print(L2)
    
    save_graph(out, "graph")
    
    print((L2 - L1)/h)
    
    
lol()
