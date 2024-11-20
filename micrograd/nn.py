import random
from engine import Value

class Neuron:
    def __init__(self, nin, label=''):
        self.w = [Value(random.uniform(-1, 1), _label=f'w{i}{label}') for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), _label=f'b{label}')

    def __call__(self, x):
        act = sum((xi*wi for xi,wi in zip(x, self.w)), self.b)
        act._label = 'act'
        out = act.tanh()
        out._label = 'out'
        return out
    
class Layer:
    def __init__(self, nin, nout, label=''):
        self.neurons = [Neuron(nin, label=f'_n{i}_L{label}') for i in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs 
    
class MLP:
    def __init__(self, sz):
        self.layers = [Layer(sz[i], sz[i+1], label=str(i)) for i in range(len(sz)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
