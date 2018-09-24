# system
import numpy as np
import argparse

# read list at specified index, if larger index is supplied, return the last value
def index_padded(iter, idx):
    idx = min(len(iter) - 1, idx)
    return iter[idx]
    
# moves the targets by the 'timedelay' indices to the right
# it is implemented by padding 'x' and 'y' on the right / left end respectively using the edge values
# if 'timedelay' is negative, the dataset is modified as a whole, otherwise each sequence is modified separately
def apply_time_delay(x, y, offsets, timedelay):
    if timedelay < 0: # shift the datasets as a whole
        x_ = np.pad(x, ((0,-timedelay),(0,0)), "edge")
        if y is not None:
            y_ = np.pad(y, ((-timedelay,0)), "edge")
        else:
            y_ = None
        if offsets is not None:
            offsets_ = offsets.copy()
            offsets_[-1] = len(x)
        else:
            offsets_ = None
    elif timedelay > 0: # shift each utterance separately
        newdatalen = x.shape[0] + (len(offsets) - 1) * timedelay
        x_ = np.zeros((newdatalen, x.shape[1]), dtype=np.float32)
        if y is not None:
            y_ = np.zeros(newdatalen, dtype=np.int32)
        else:
            y_ = None
        offsets_ = offsets.copy()
        ptr = 0
        for o in range(len(offsets_) - 1):
            l = offsets_[o+1] - offsets_[o]
            nextptr = ptr + l + timedelay
            x_[ptr:nextptr] = np.pad(x[offsets_[o]:offsets_[o+1]], ((0,timedelay),(0,0)), "edge")
            if y is not None:
                y_[ptr:nextptr] = np.pad(y[offsets_[o]:offsets_[o+1]], ((timedelay,0)), "edge")
            offsets_[o] = ptr
            ptr = nextptr
        offsets_[-1] = ptr
    return x_, y_, offsets_

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')