# system
import numpy as np

def loadKaldiFeatureTransform(filename):
    s = open(filename).readlines()
    ft = {}
    ft["shape"] = [int(foo) for foo in s[1].split()[1:]]
    ft["shifts"] = [int(foo) for foo in s[2].split()[1:-1]]
    ft["addShift"] = np.reshape(np.asarray([float(foo) for foo in s[4].split()[3:-1]], dtype=np.float32), [1, ft["shape"][0]])[0]
    ft["rescale"]  = np.reshape(np.asarray([float(foo) for foo in s[6].split()[3:-1]], dtype=np.float32), [1, ft["shape"][0]])[0]
    return ft

def applyKaldiFeatureTransform(x, ft):
    x1 = (x + ft["addShift"])
    x2 = x1 * ft["rescale"]
    #x = (x + ft["addShift"]) * ft["rescale"]
    return x2

def prepareBatch(data, targets, idxs, winlen):
    winhalf = int(winlen/2)
    batchSize = len(idxs)
    num, dim = data.shape
    x = np.zeros([batchSize, winlen * dim], dtype=np.float32)
    x1 = np.zeros([winlen * dim], dtype=np.float32)
    idxs.sort()
    for k, idx in enumerate(idxs):
        if -winhalf+idx < 0 or  winhalf+idx >= num:
            for wi, w in enumerate(range(-winhalf+idx, winhalf+1+idx)):
                if w < 0:
                    w = 0
                if w >= num:
                    w = num-1
                x1[wi*dim:(wi+1)*dim] = data[w]
            x[k] = x1
        else:
            x[k] = data[idx-winhalf:winhalf+1+idx].flatten()

    if targets != []:
        target = targets[idxs]
    else:
        target = []

    return x, target
