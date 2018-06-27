# system
from pathlib import Path
import os
import progressbar
import subprocess
import numpy as np
import chainer
from chainer.cuda import to_gpu, to_cpu
# util
from kw_utils import saveBin, logsum, loadMlf
from levenshtein import computeWER

def convert_results(file_phone_map, file_in, file_out):

    #read phone mapping
    phone_map = {}
    with open(file_phone_map) as fid:
        for p in fid.readlines():
            p = p.strip()
            ps = p.split()
            if len(ps) == 3:
                phone_map[ps[1]] = ps[2]

    #read mlf, and do mapping
    with open(file_out, 'w') as fid:
        for p in open(file_in).readlines():
            p = p.strip()
            if p and p[0] == '"':
                p = p.replace('.lab', '.rec').replace('\\', '/')
            ps = p.split()
            if ps and not ps[0][0] in ['#', '.', '"']:
                ps[-1] = phone_map[ps[-1]]
                p = ' '.join(ps)
            fid.write(p + '\n')

def evaluateModelTestTri(model, data, offsets, PIP, LMW, ap=None, GPUID=-1, testOrDev='test', tmpDir='lab', uttlistdir=".", recogdir=".", progress=True, rnn=False):

    testList = open(uttlistdir + '/' + testOrDev + '.list').readlines()
    if len(testList) != len(offsets) - 1:
        print("Error: wrong number of utterances")
        return -1

    labDirOut = Path(tmpDir)
    labDirOut.mkdir(exist_ok=True, parents=True)
    fscp = open(str(Path(labDirOut, testOrDev + '.scp')), 'wt')
    chainer.config.train = False
    if GPUID < 0:
        model.to_cpu()
    else:
        model.to_gpu(device=GPUID)

    if rnn:
        testListLen = np.diff(offsets)
        testListIdx = np.flip(testListLen.argsort(), axis=0)
        testListIdxRev = np.zeros(len(testList), dtype=np.int)
        testListIdxRev[testListIdx] = range(len(testListIdx))
        
        xb = np.zeros((len(testList), testListLen[testListIdx[0]], data.shape[1]), dtype=np.float32)
        for i, idx in enumerate(testListIdx):
            xb[i, :testListLen[idx], :] = data[offsets[idx]:offsets[idx + 1], :]

        print("Calculating network outputs")
        yb = None
        model.reset_state()
        if progress:
            bar = progressbar.ProgressBar(max_value=xb.shape[1])
        for t in range(xb.shape[1]):
            batch_size = np.sum(testListLen > t)
            xg = xb[:, t, :]
            if GPUID >= 0:
                xg = to_gpu(xg, device=GPUID)
            with chainer.no_backprop_mode():
                y = model(chainer.Variable(xg)).data
            y = to_cpu(y)
            if ap is not None:
                y = y - ap
            y = y - logsum(y, axis=1)
            if yb is None:
                yb = np.zeros((xb.shape[0], xb.shape[1], y.shape[1]), dtype=np.float32)
            yb[:batch_size, t, :] = y[:batch_size]

            if progress:
                bar += 1
        if progress:
            print()

        print("Writing output files")
        for i, f in enumerate(testList):
            idx = testListIdxRev[i]
            f = f.strip()
            labout = Path(labDirOut, f + '.lab')
            saveBin(str(labout), yb[idx, :testListLen[i], :])
            fscp.write(str(labout) + '\n')
        fscp.close()
    else:
        if progress:
            bar = progressbar.ProgressBar(max_value=len(testList))
        for i, f in enumerate(testList):
            f = f.strip()
            dataW = data[offsets[i]:offsets[i + 1], :]
            
            if GPUID < 0:
                dataG = dataW
            else:
                dataG = to_gpu(dataW, device=GPUID)
            with chainer.no_backprop_mode():
                y = model(dataG).data
            y = to_cpu(y)
                
            if ap is not None:
                y = y - ap
            y = y - logsum(y, axis=1)
            
            labout = Path(labDirOut, f + '.lab')
            saveBin(str(labout), y)
            fscp.write(str(labout) + '\n')

            if progress:
                bar += 1
        fscp.close()
        if progress:
            print()
      
    cmd = [recogdir + '/PhoneRecog', str(Path(labDirOut, testOrDev + '.scp')), str(Path(recogdir, 'kaldiTri1909.img')), str(Path(labDirOut, 'vysledek_' + testOrDev + '.txt')), str(-abs(PIP)), str(LMW)]
    if os.name == "nt":
        cmd[0] += ".exe"
    subprocess.run(cmd, cwd=os.getcwd())
    convert_results(str(Path(recogdir, 'phones.60-48-39.map')), str(Path(labDirOut, 'vysledek_' + testOrDev + '.txt')), str(Path(labDirOut, 'vysledek_' + testOrDev + '_p39.txt')))
    
    ref = loadMlf(str(Path(recogdir, testOrDev + '_ref.mlf')))
    result = loadMlf(str(Path(labDirOut, 'vysledek_' + testOrDev + '_p39.txt')))
    wer = computeWER(result, ref, True)
    return wer