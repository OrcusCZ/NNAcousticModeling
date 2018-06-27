# system
import chainer
import chainer.functions as F
import chainer.links as L
# common
from MGRU import MGRU

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out, layers=2, dropout=0, activation=F.relu):
        super(MLP, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        with self.init_scope():
            for l in range(layers):
                setattr(self, "layer_{}".format(l), L.Linear(None, n_units))
            self.out = L.Linear(None, n_out)

    def __call__(self, x):
        for l in range(self.layers):
            x = F.dropout(self.activation(getattr(self, "layer_{}".format(l))(x)), self.dropout)
        return self.out(x)
        
class TDNN(chainer.Chain):
    def __init__(self, n_units, n_out, ksize, dropout=0, activation=F.relu):
        super(TDNN, self).__init__()
        if len(n_units) != len(ksize):
            raise ValueError("TDNN n_units argument must have the same length as ksize")
        self.layers = len(n_units)
        self.dropout = dropout
        self.activation = activation
        self.input_win_size = sum(ksize) - len(ksize) + 1
        with self.init_scope():
            for l, units in enumerate(n_units):
                setattr(self, "layer_{}".format(l), L.Convolution2D(None, units, ksize=(1, ksize[l]), stride=1))
            self.out = L.Linear(None, n_out)

    def __call__(self, x):
        x = F.reshape(x, (x.shape[0], -1, 1, self.input_win_size))
        for l in range(self.layers):
            x = F.dropout(self.activation(getattr(self, "layer_{}".format(l))(x)), self.dropout)
        return self.out(x)

class LSTM(chainer.Chain):
    def __init__(self, n_units, n_out, layers=2, dropout=0):
        super(LSTM, self).__init__()
        self.layers = layers
        self.dropout = dropout
        with self.init_scope():
            for l in range(layers):
                setattr(self, "layer_{}".format(l), L.LSTM(None, n_units))
            self.out = L.Linear(None, n_out)
    
    def reset_state(self):
        for l in range(self.layers):
            getattr(self, "layer_{}".format(l)).reset_state()

    def __call__(self, x):
        x = F.dropout(x, self.dropout)
        for l in range(self.layers):
            x = F.dropout(getattr(self, "layer_{}".format(l))(x), self.dropout)
        return self.out(x)

class ZoneoutLSTM(chainer.Chain):
    def __init__(self, n_units, n_out, layers=2, c_ratio=0.5, h_ratio=0.5):
        super(ZoneoutLSTM, self).__init__()
        self.layers = layers
        with self.init_scope():
            for l in range(layers):
                setattr(self, "layer_{}".format(l), L.StatefulZoneoutLSTM(None, n_units, c_ratio, h_ratio))
            self.out = L.Linear(None, n_out)
    
    def reset_state(self):
        # print("reset_state called")
        for l in range(self.layers):
            getattr(self, "layer_{}".format(l)).reset_state()

    def __call__(self, x):
        for l in range(self.layers):
            x = getattr(self, "layer_{}".format(l))(x)
        return self.out(x)

class ZoneoutDropoutLSTM(chainer.Chain):
    def __init__(self, n_units, n_out, layers=2, dropout=0, c_ratio=0.5, h_ratio=0.5):
        super(ZoneoutDropoutLSTM, self).__init__()
        self.layers = layers
        self.dropout = dropout
        with self.init_scope():
            for l in range(layers):
                setattr(self, "layer_{}".format(l), L.StatefulZoneoutLSTM(None, n_units, c_ratio, h_ratio))
            self.out = L.Linear(None, n_out)
    
    def reset_state(self):
        for l in range(self.layers):
            getattr(self, "layer_{}".format(l)).reset_state()

    def __call__(self, x):
        x = F.dropout(x, self.dropout)
        for l in range(self.layers):
            x = F.dropout(getattr(self, "layer_{}".format(l))(x), self.dropout)
        return self.out(x)

class PeepholeLSTM(chainer.Chain):
    def __init__(self, n_units, n_out, layers=2, dropout=0):
        super(PeepholeLSTM, self).__init__()
        self.layers = layers
        self.dropout = dropout
        with self.init_scope():
            for l in range(layers):
                setattr(self, "layer_{}".format(l), L.StatefulPeepholeLSTM(None, n_units))
            self.out = L.Linear(None, n_out)
    
    def reset_state(self):
        for l in range(self.layers):
            getattr(self, "layer_{}".format(l)).reset_state()

    def __call__(self, x):
        x = F.dropout(x, self.dropout)
        for l in range(self.layers):
            x = F.dropout(getattr(self, "layer_{}".format(l))(x), self.dropout)
        return self.out(x)

class GRU(chainer.Chain):
    def __init__(self, n_units, n_out, layers=2, dropout=0):
        super(GRU, self).__init__()
        self.layers = layers
        self.dropout = dropout
        with self.init_scope():
            for l in range(layers):
                setattr(self, "layer_{}".format(l), L.GRU(None, n_units))
            self.out = L.Linear(None, n_out)
    
    def reset_state(self):
        for l in range(self.layers):
            getattr(self, "layer_{}".format(l)).reset_state()

    def __call__(self, x):
        x = F.dropout(x, self.dropout)
        for l in range(self.layers):
            x = F.dropout(getattr(self, "layer_{}".format(l))(x), self.dropout)
        return self.out(x)

class NetMGRU(chainer.Chain):
    def __init__(self, n_units, n_out, layers=2, dropout=0, use_reset_gate=False, activation=F.relu):
        super(NetMGRU, self).__init__()
        self.layers = layers
        self.dropout = dropout
        with self.init_scope():
            for l in range(layers):
                setattr(self, "layer_{}".format(l), MGRU(None, n_units, use_reset_gate=use_reset_gate, activation=activation))
            self.out = L.Linear(None, n_out)
    
    def reset_state(self):
        for l in range(self.layers):
            getattr(self, "layer_{}".format(l)).reset_state()

    def __call__(self, x):
        x = F.dropout(x, self.dropout)
        for l in range(self.layers):
            x = F.dropout(getattr(self, "layer_{}".format(l))(x), self.dropout)
        return self.out(x)

def get_nn(network, layers, units, num_classes, activation, tdnn_ksize, dropout=[0]):
    if network == "ff":
        return MLP(units[0], num_classes, layers, dropout[0], activation)
    elif network == "tdnn":
        return TDNN(units, num_classes, tdnn_ksize, dropout[0], activation)
    elif network == "lstm":
        return LSTM(units[0], num_classes, layers, dropout[0])
    elif network == "zoneoutlstm":
        return ZoneoutLSTM(units[0], num_classes, layers, *dropout)
    elif network == "zoneoutdropoutlstm":
        return ZoneoutDropoutLSTM(units[0], num_classes, layers, *dropout)
    elif network == "peepholelstm":
        return PeepholeLSTM(units[0], num_classes, layers, dropout[0])
    elif network == "gru":
        return GRU(units[0], num_classes, layers, dropout[0])
    elif network == "mgrurelu":
        return NetMGRU(units[0], num_classes, layers, dropout[0], False, F.relu)
    elif network == "mgrurelur":
        return NetMGRU(units[0], num_classes, layers, dropout[0], True, F.relu)
    else:
        print("Wrong network type specified")
        exit(1)

def is_nn_recurrent(n):
    return n.endswith("lstm") or n.startswith("gru") or n.startswith("mgru")
