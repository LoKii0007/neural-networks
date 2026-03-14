from viz import save_graph
import math

class Value:
    
    def __init__(self, data, _chidren = (), _op = "", label="", grad = 0.0):
        self.data = data
        self._prev = set(_chidren)
        self._op = _op
        self.label = label
        self.grad = grad
        self._backward = lambda:None
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def _ensure_value(self, other):
        return other if isinstance(other, Value) else Value(other)
        
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        
        out = Value(self.data ** other, (self,), "pow")
        
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
            
        out._backward = _backward
        
        return out
        
    def __neg__(self):
        return self * -1.0
        
    def __add__(self, other):
        other = self._ensure_value(other)
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = self._ensure_value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + (-other)

    
    def __rsub__(self, other):
        return self - other
    
    def __truediv__(self, other):
        return self * other ** (-1)
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
            
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        tanx = (math.exp(2*x) -1)/(math.exp(2*x) + 1)
        out = Value(tanx, (self, ), "tan" )
        
        def _backward():
            self.grad += (1 - tanx ** 2) * out.grad
            
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def lol():
    
    h = 0.001
    
    a = Value(data=2.0, label = "a")
    b = Value(data=-3.0, label = "b")
    e = a * b
    e.label = "e"
    c = Value(data=10.0, label = "c")
    d = c + e
    d.label = "d"
    f = Value(data=-2.0, label = "f")
    L = f * d
    L.label = "L"
    L1 = L.data
    print(L1)
    
    
    a = Value(data=2.0 + h, label = "a")
    b = Value(data=-3.0, label = "b")
    e = a * b
    e.label = "e"
    c = Value(data=10.0, label = "c")
    d = c + e
    d.label = "d"
    f = Value(data=-2.0, label = "f")
    L = f * d
    L.label = "L"
    L2 =L.data
    print(L2)
    
    L.grad = 1.0
    
    # backpropagationg the derivates - 
    # mul operation
    # dl/dd = f
    f.grad = d.data
    d.grad = f.data
    
    # addition op - addition is 1.0
    c.grad = d.grad
    e.grad = d.grad
    
    # mul op
    a.grad = b.data * e.grad
    b.grad = a.data * e.grad
    
    save_graph(L, "graph")
    
    print((L2 - L1)/h)
    
    
# lol()

def visual_recognition():
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    
    # weights 
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    
    x1w1 = x1 * w1; x1w1.label = "x1w1"
    x2w2 = x2 * w2; x2w2.label = "x2w2"
    
    Sxiwi = x1w1 + x2w2; Sxiwi.label = "submation"
    
    
    # bias 
    b = Value(6.8813735870195432, label="b")
    
    n = Sxiwi + b ; n.label = "n"
    o = n.tanh(); o.label = "o"
    print(o)
    
    # backpropagationg derrivates 
    o.grad = 1.0
    
    # o = tanh 
    # do/dn = 1 - tanx **2
    n.grad = 1- o.data **2
    
    # we have just + op so derivate is just flowing normally
    b.grad = n.grad
    Sxiwi.grad = n.grad
    
    # same we have just + operation again
    x1w1.grad = Sxiwi.grad
    x2w2.grad = Sxiwi.grad
    
    # now we have multiplcation operation
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad
    
    save_graph(o, "graph-2")
    
# visual_recognition()

def visual_recognition_2():
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    
    # weights 
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    
    x1w1 = x1 * w1; x1w1.label = "x1w1"
    x2w2 = x2 * w2; x2w2.label = "x2w2"
    
    Sxiwi = x1w1 + x2w2; Sxiwi.label = "submation"
    
    
    # bias 
    b = Value(6.8813735870195432, label="b")
    
    n = Sxiwi + b ; n.label = "n"
    # o = n.tanh(); o.label = "o"
    
    e = (2*n).exp()
    o= (e-1)/(e+1) 
    print(o)
    
    # backpropagationg derrivates using function
    # o.grad = 1.0
    # 
    # o._backward()
    # n._backward()
    # b._backward()
    # Sxiwi._backward()
    # x1w1._backward()
    # x2w2._backward()
    
    o.backward()
    
    save_graph(o, "graph-2")

# visual_recognition_2()

def testing():
   a = Value(2.0, label="a")
   b = Value(1.0, label="b")
   ans = a/b
   print(ans)  
   
# testing()