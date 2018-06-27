#!/usr/bin/env python3

# system
import os
import sys
import argparse
import progressbar
from pathlib import Path
import random
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, serializers
from chainer.training import extensions, StandardUpdater
from chainer.iterators import SerialIterator
# common
from chainer_networks import get_nn, is_nn_recurrent
from RPL import RPL0, RPL, RPL2, RPL3, RPL4
# util
from orcus_util import index_padded, apply_time_delay
from orcus_chainer_util import model_saver, SequenceShuffleIterator, BPTTUpdater
from kw_utils import loadBin, saveBin, logsum
from kw_nn_utils import loadKaldiFeatureTransform, applyKaldiFeatureTransform, prepareBatch
from chainer_kw_utils import EarlyStoppingTrigger

def predict(model, x, offsets, num_classes, network, gpu, winlen, timedelay, ft, progress=True):
    if is_nn_recurrent(network):
        utt_len = offsets[1:] - offsets[:-1]
        utt_idx = np.flip(utt_len.argsort(), axis=0)
        utt_idx_rev = np.zeros(len(utt_len), dtype=np.int)
        utt_idx_rev[utt_idx] = range(len(utt_len))

        xb = np.zeros((len(utt_len), utt_len[utt_idx[0]] + timedelay, x.shape[1]), dtype=np.float32)
        for i, idx in enumerate(utt_idx):
            offset_beg = offsets[idx]
            offset_end = offsets[idx + 1]
            x_ = x[offset_beg:offset_end, :]
            x_ = np.pad(x_, ((0,timedelay),(0,0)), mode="edge")
            if ft is not None:
                xb[i, :x_.shape[0], :] = applyKaldiFeatureTransform(x_, ft)
            else:
                xb[i, :x_.shape[0], :] = x_

        yb = None
        model.reset_state()
        if progress:
            bar = progressbar.ProgressBar(max_value=xb.shape[1])
        for t in range(xb.shape[1]):
            batch_size = np.sum(utt_len > t)
            xg = xb[:, t, :]
            if gpu >= 0:
                xg = chainer.cuda.to_gpu(xg, device=gpu)
            with chainer.no_backprop_mode():
                y = model(chainer.Variable(xg)).data
            y = chainer.cuda.to_cpu(y)
            y = y - logsum(y, axis=1)
            if yb is None:
                yb = np.zeros((xb.shape[0], xb.shape[1] - timedelay, y.shape[1]), dtype=np.float32)
            if t >= timedelay:
                yb[:batch_size, t - timedelay, :] = y[:batch_size]

            if progress:
                bar += 1
        y_out = []
        for i, idx in enumerate(utt_idx_rev):
            y_out.append(yb[idx, :utt_len[i], :].reshape((utt_len[i], -1)))
        y_out = np.concatenate(y_out, axis=0)
    else:
        y_out = []
        batch_size = 1024
        offset = 0
        
        if progress:
            bar = progressbar.ProgressBar(max_value=x.shape[0])
        while offset < x.shape[0]:
            offset_end = min(offset + batch_size, x.shape[0])
            x_, _ = prepareBatch(x, [], np.arange(offset, offset_end), winlen)
            if ft is not None:
                x_ = applyKaldiFeatureTransform(x_, ft)
            if gpu < 0:
                xg = x_
            else:
                xg = chainer.cuda.to_gpu(x_, device=gpu)
            with chainer.no_backprop_mode():
                y = model(xg).data
            y = chainer.cuda.to_cpu(y)
            y = y - logsum(y, axis=1)
            y_out.append(y)
            offset += batch_size
            if progress:
                bar.update(offset_end)
            
        y_out = np.concatenate(y_out, axis=0)
    return y_out

def main(arg_list=None):
    parser = argparse.ArgumentParser(description='Chainer LSTM')
    parser.add_argument('--network', '-n', default='ff', help='Neural network type, either "ff", "lstm" or "tdnn". Setting "lstm" implies "--shuffle-sequences"')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--units', '-u', type=int, nargs='+', default=[1024], help='Number of units')
    parser.add_argument('--layers', '-l', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--activation', '-a', default='relu', help='FF activation function (sigmoid, tanh or relu)')
    parser.add_argument('--tdnn-ksize', type=int, nargs='+', default=[5], help='TDNN kernel size')
    parser.add_argument('--timedelay', type=int, default=0, help='Delay target values by this many time steps')
    parser.add_argument('--splice', type=int, default=0, help='Splicing size')
    parser.add_argument('--dropout', '-d', type=float, nargs='+', default=0, help='Dropout rate (0 to disable). In case of Zoneout LSTM, this parameter has 2 arguments: c_ratio h_ratio')
    parser.add_argument('--ft', help='Kaldi feature transform file')
    parser.add_argument('--tri', action='store_true', help='Use triphones')
    parser.add_argument('--data-dir', default='data/fmllr')
    parser.add_argument('--offset-dir', default='data')
    parser.add_argument('--ivector-dir', help="Directory with i-vector files")
    parser.add_argument('--data', default='data_{}.npy', help='Input data')
    parser.add_argument('--offsets', default='offsets_{}.npy', help='Input offsets')
    parser.add_argument('--ivectors', default='ivectors_{}.npy')
    parser.add_argument('--fold-data-dir', help='Directory with fold input data')
    parser.add_argument('--fold-output-dir', help='Directory with predicted fold output')
    parser.add_argument('--fold-model-dir', help='Directory with output fold model')
    parser.add_argument('--fold-output-dev', help='Output file with predicted development data values')
    parser.add_argument('--fold-data-pattern', default='data_{}.npy', help='Filename pattern of each fold data, {} will be replaced by fold ID')
    parser.add_argument('--fold-offset-pattern', default='offsets_{}.npy', help='Filename pattern of each fold offset')
    parser.add_argument('--fold-ivector-pattern', default='ivectors_{}.npy', help='Filename pattern of each fold i-vectors file, {} will be replaced by fold ID')
    parser.add_argument('--fold-output-pattern', default='data_{}.npy', help='Filename pattern of each fold network output')
    parser.add_argument('--fold-network-pattern', default='fold_{}.npz', help='Filename pattern of each fold network')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    if arg_list is not None:
        args = parser.parse_args(list(map(str, arg_list)))
    else:
        args = parser.parse_args()

    # create output directories
    if args.fold_output_dev is not None:
        out_file = Path(args.fold_output_dir, args.fold_output_dev)
    else:
        out_file = Path(args.fold_output_dir, args.fold_output_pattern)
    Path(out_file.parent).mkdir(exist_ok=True, parents=True)

    # input feature vector length
    num_classes = 1909 if args.tri else 39
    
    chainer.config.train = False
    
    # create model
    if args.activation == "sigmoid":
        activation = F.sigmoid
    elif args.activation == "tanh":
        activation = F.tanh
    elif args.activation == "relu":
        activation = F.relu
    else:
        print("Wrong activation function specified")
        exit(1)
    model = get_nn(args.network, args.layers, args.units, num_classes, activation, args.tdnn_ksize, args.dropout)

    # classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model_cls = L.Classifier(model)
    if args.gpu >= 0:
        # make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model_cls.to_gpu()  # copy the model to the GPU
        
    if args.network == "tdnn":
        splice = (sum(args.tdnn_ksize) - len(args.tdnn_ksize)) // 2
    else:
        splice = args.splice
    winlen = 2 * splice + 1
        
    # load feature transform
    if not args.ft and args.ft != '-':
        ft = loadKaldiFeatureTransform(str(Path(args.data_dir, args.ft)))
        if is_nn_recurrent(args.network): # select transform middle frame if the network is recurrent
            dim = ft["shape"][1]
            zi = ft["shifts"].index(0)
            ft["rescale"] = ft["rescale"][zi*dim:(zi+1)*dim]
            ft["addShift"] = ft["addShift"][zi*dim:(zi+1)*dim]
            ft["shape"][0] = dim
            ft["shifts"] = [0]
        elif args.network == "tdnn":
            dim = ft["shape"][1]
            zi = ft["shifts"].index(0)
            winlen = 2 * splice + 1
            ft["rescale"] = np.tile(ft["rescale"][zi*dim:(zi+1)*dim], winlen)
            ft["addShift"] = np.tile(ft["addShift"][zi*dim:(zi+1)*dim], winlen)
            ft["shape"][0] = dim * winlen
            ft["shifts"] = list(range(-splice, splice + 1))
    else:
        ft = None
            
    if args.fold_output_dev is not None:
        x = np.load(str(Path(args.data_dir, args.data.format("dev"))))
        if is_nn_recurrent(args.network):
            offsets = np.load(str(Path(args.offset_dir, args.offsets.format("dev"))))
        else:
            offsets = None
        if args.ivector_dir:
            ivectors = np.load(str(Path(args.ivector_dir, args.ivectors.format("dev"))))
            x = np.concatenate((x, ivectors), axis=1)
        y_out = 0
        
        fold = 0
        while True:
            model_file = Path(args.fold_model_dir, args.fold_network_pattern.format(fold))
            if not model_file.is_file():
                break
            serializers.load_npz(str(model_file), model_cls)
            print("Predicting fold {} data".format(fold))
            
            y = predict(model, x, offsets, num_classes, args.network, args.gpu, winlen, args.timedelay, ft, not args.no_progress)   
            y_out += y
            
            fold += 1
                
        if fold == 0:
            print("Error: No fold networks found")
            exit(2)
            
        y_out /= fold
        y_out = y_out - logsum(y_out, axis=1)
        np.save(str(Path(args.fold_output_dir, args.fold_output_dev)), y_out)
    else:
        fold = 0
        while True:
            model_file = Path(args.fold_model_dir, args.fold_network_pattern.format(fold))
            if not model_file.is_file():
                break
            serializers.load_npz(str(model_file), model_cls)
            print("Predicting fold {} data".format(fold))

            x = np.load(str(Path(args.fold_data_dir, args.fold_data_pattern.format(fold))))
            if is_nn_recurrent(args.network):
                offsets = np.load(str(Path(args.fold_data_dir, args.fold_offset_pattern.format(fold))))
            else:
                offsets = None
            if args.ivector_dir:
                ivectors = np.load(str(Path(args.fold_data_dir, args.fold_ivector_pattern.format(fold))))
                x = np.concatenate((x, ivectors), axis=1)
                        
            y = predict(model, x, offsets, num_classes, args.network, args.gpu, winlen, args.timedelay, ft, not args.no_progress)   
            np.save(str(Path(args.fold_output_dir, args.fold_output_pattern.format(fold))), y)
            
            fold += 1
                
        if fold == 0:
            print("Error: No fold networks found")
            exit(2)
           
if __name__ == '__main__':
    main()
