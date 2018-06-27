# system
import numpy
from chainer.functions.activation import sigmoid, tanh, relu
from chainer.functions.math import linear_interpolate
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class MGRUBase(link.Chain):

    def __init__(self, in_size, out_size, init=None,
                 inner_init=None, bias_init=None, use_reset_gate=True):
        super(MGRUBase, self).__init__()
        self.use_reset_gate = use_reset_gate
        with self.init_scope():
            if use_reset_gate:
                self.W_r = linear.Linear(
                    in_size, out_size, initialW=init, initial_bias=bias_init)
                self.U_r = linear.Linear(
                    out_size, out_size, initialW=inner_init,
                    initial_bias=bias_init)
            self.W_z = linear.Linear(
                in_size, out_size, initialW=init, initial_bias=bias_init)
            self.U_z = linear.Linear(
                out_size, out_size, initialW=inner_init,
                initial_bias=bias_init)
            self.W = linear.Linear(
                in_size, out_size, initialW=init, initial_bias=bias_init)
            self.U = linear.Linear(
                out_size, out_size, initialW=inner_init,
                initial_bias=bias_init)


class StatefulMGRU(MGRUBase):

    def __init__(self, in_size, out_size, init=None,
                 inner_init=None, bias_init=0, use_reset_gate=True, activation=tanh.tanh):
        super(StatefulMGRU, self).__init__(
            in_size, out_size, init, inner_init, bias_init, use_reset_gate)
        self.state_size = out_size
        self.activation = activation
        self.reset_state()

    def to_cpu(self):
        super(StatefulMGRU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulMGRU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, variable.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            z += self.U_z(self.h)
            if self.use_reset_gate:
                r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
                h_bar += self.U(r * self.h)
            else:
                h_bar += self.U(self.h)
        z = sigmoid.sigmoid(z)
        h_bar = self.activation(h_bar)

        if self.h is not None:
            h_new = linear_interpolate.linear_interpolate(z, h_bar, self.h)
        else:
            h_new = z * h_bar
        self.h = h_new
        return self.h


class MGRU(StatefulMGRU):

    def __call__(self, *args):
        n_args = len(args)
        msg = ("Invalid argument. The length of MGRU.__call__ must be 1. "
               "But %d is given. " % n_args)

        if n_args == 0 or n_args >= 3:
            raise ValueError(msg)
        elif n_args == 2:
            msg += ("MGRU is stateful. "
                    "One possiblity is you assume MGRU to be stateless. "
                    "Stateless MGRU is not supported.")
            raise ValueError(msg)

        return super(MGRU, self).__call__(args[0])
