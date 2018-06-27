# system
import chainer
from chainer import Chain, initializers, variable
import chainer.functions as F
import chainer.links as L
import numpy as np

class RPL0(Chain):
    def __init__(self, n_out):
        super(RPL0, self).__init__()

    def __call__(self, x):
        return x

class RPL(Chain):
    def __init__(self, n_out):
        super(RPL, self).__init__()
        with self.init_scope():
            self.l = L.Linear(None, n_out)

    def __call__(self, x):
        x = F.log_softmax(x)
        y = x + self.l(x)
        return y

class RPL2(Chain):
    def __init__(self, n_out):
        super(RPL2, self).__init__()
        with self.init_scope():
            self.l = L.Linear(None, n_out, initialW=np.zeros((n_out, n_out), dtype=np.float32), initial_bias=np.zeros((n_out), dtype=np.float32))
            logbias_initializer = initializers.Constant(-20.0)
            self.lb = variable.Parameter(logbias_initializer, (1, n_out))

    def __call__(self, x):
        x = F.log_softmax(x)
        h = x + self.l(x)
        mx = F.maximum(h, F.broadcast_to(self.lb, x.shape))
        mn = F.minimum(h, F.broadcast_to(self.lb, x.shape))
        y = mx + F.log(1.0 + F.exp(mn - mx))
        return y

class RPL3(Chain):
    def __init__(self, n_out):
        super(RPL3, self).__init__()
        with self.init_scope():
            self.l = L.Linear(None, n_out, initialW=np.zeros((n_out, n_out), dtype=np.float32), initial_bias=np.zeros((n_out), dtype=np.float32))
            logbias_initializer = initializers.Constant(-20.0)
            self.lb = variable.Parameter(logbias_initializer, (1, n_out))

    def __call__(self, x):
        x = F.log_softmax(x)
        h = x + self.l(x)
        mx = F.maximum(h, F.broadcast_to(self.lb, x.shape))
        mn = F.minimum(h, F.broadcast_to(self.lb, x.shape))
        y = mx + F.log(1.0 + F.exp(mn - mx))
        return y

class RPL4(Chain):
    def __init__(self, n_out):
        super(RPL4, self).__init__()
        with self.init_scope():
            zero_initializer = initializers.Constant(0.0)
            self.W = variable.Parameter(zero_initializer, (1, n_out))
            self.b = variable.Parameter(zero_initializer, (1, n_out))
            logbias_initializer = initializers.Constant(-20.0)
            self.lb = variable.Parameter(logbias_initializer, (1, n_out))

    def __call__(self, x):
        x = F.log_softmax(x)
        h = x + x * F.broadcast_to(self.W, x.shape) + F.broadcast_to(self.b, x.shape)
        mx = F.maximum(h, F.broadcast_to(self.lb, x.shape))
        mn = F.minimum(h, F.broadcast_to(self.lb, x.shape))
        y = mx + F.log(1.0 + F.exp(mn - mx))
        return y
