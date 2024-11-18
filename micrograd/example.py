from engine import Value
from graph import draw_dot


x1 = Value(2, _label='x1')
x2 = Value(0, _label='x2')

w1 = Value(-3, _label='w1')
w2 = Value(1, _label='w2')

b = Value(6.8813735870195432, _label='b')

x1w1 = x1*w1
x1w1._label = 'x1w1'
x2w2 = x2*w2
x2w2._label = 'x2w2'

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2._label = 'x1w1 + x2w2'
n = x1w1x2w2 + b
n._label = 'n'
# o = n.tanh();
e = (2*n).exp()
o = (e-1) / (e+1)
o._label = 'o'

o.backward()
dot = draw_dot(o)
dot.render('assets/computation_graph', format='svg', view=True)
