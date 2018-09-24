#!/usr/bin/env python3

# system
import sys
import argparse
from pathlib import Path
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
# common
from chainer_networks import get_nn, is_nn_recurrent
from RPL import RPL0, RPL, RPL2, RPL3, RPL4
# util
from kw_utils import splicing
from kw_nn_utils import loadKaldiFeatureTransform, applyKaldiFeatureTransform
from evaluateModelForTest import evaluateModelTestTri

class NNWithRPL(chainer.Chain):
    def __init__(self, master=None, folds=[], rpl=None):
        super(NNWithRPL, self).__init__()
        self.num_folds = len(folds)
        with self.init_scope():
            self.master = master
            for i in range(self.num_folds):
                setattr(self, "fold_{}".format(i), folds[i])
            self.rpl = rpl

    def reset_state(self):
        if self.master is not None:
            self.master.reset_state()
        for i in range(self.num_folds):
            getattr(self, "fold_{}".format(i)).reset_state()
                
    def __call__(self, x):       
        if self.master is not None and self.num_folds == 0:
            h = self.master(x)
        elif self.master is not None:
            h = self.master(x) * self.num_folds
            for i in range(self.num_folds):
                h += getattr(self, "fold_{}".format(i))(x)
            h /= 2 * self.num_folds
        else:
            h = 0
            for i in range(self.num_folds):
                h += getattr(self, "fold_{}".format(i))(x)
            h /= self.num_folds
        
        if self.rpl is not None:
            h = self.rpl(h)
        return h

def main(arg_list=None):
    parser = argparse.ArgumentParser(description='Chainer Evaluation')
    parser.add_argument('--network', '-n', default='ff', help='Neural network type, either "ff" or "lstm"')
    parser.add_argument('--model', '-m', default='', help='Path to the model')
    parser.add_argument('--units', '-u', type=int, nargs='+', default=[1024], help='Number of units')
    parser.add_argument('--layers', '-l', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--activation', '-a', default='relu', help='FF activation function (sigmoid, tanh or relu)')
    parser.add_argument('--tdnn-ksize', type=int, nargs='+', default=[5], help='TDNN kernel size')
    parser.add_argument('--timedelay', type=int, default=0, help='Delay target values by this many time steps')
    parser.add_argument('--splice', type=int, default=0, help='Splicing size')
    parser.add_argument('--dropout', '-d', type=float, nargs='+', default=[0], help='Dropout rate (0 to disable). In case of Zoneout LSTM, this parameter has 2 arguments: c_ratio h_ratio')
    parser.add_argument('--tri', action='store_true', help='Use triphones')
    parser.add_argument('--ft', default='final.feature_transform', help='Kaldi feature transform file')
    parser.add_argument('--data-dir', default='data/fmllr', help='Data directory, this will be prepended to data files and feature transform')
    parser.add_argument('--offset-dir', default='data', help='Data directory, this will be prepended to offset files')
    parser.add_argument('--ivector-dir', help='Data directory, this will be prepended to ivector files')
    parser.add_argument('--recog-dir', required=True, help='Directory with recognizer files')
    parser.add_argument('--utt-list-dir', default='data', help='Directory with utterance lists')
    parser.add_argument('--data', default='data_{}.npy', help='Data file')
    parser.add_argument('--offsets', default='offsets_{}.npy', help='Offset file')
    parser.add_argument('--ivectors', default='ivectors_{}.npy', help='ivectors file')
    parser.add_argument('--PIP', type=float, default=20)
    parser.add_argument('--LMW', type=float, default=1)
    parser.add_argument('--ap-coef', type=float, default=1)
    parser.add_argument('--ap-file', default='log_ap_Kaldi1909.npy', help='Path relative to recogdir')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test-or-dev', default='test', help='Test or dev')
    parser.add_argument('--rpl', action='store_true', help='Use RPL layer with folds')
    parser.add_argument('--no-rpl-layer', action='store_true', help='Disable RPL layer')
    parser.add_argument('--rpl-model', default="result_rpl/model", help='RPL layer model')
    parser.add_argument('--fold-model-dir', default="fold_models", help='Directory with trained fold models')
    parser.add_argument('--fold-network-pattern', default='fold_{0}.npz', help='Filename pattern of each fold network')
    parser.add_argument('--master-network', default="-", help='Master network')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    if arg_list is not None:
        args = parser.parse_args(list(map(str, arg_list)))
    else:
        args = parser.parse_args()
    
    num_classes = 1909 if args.tri else 39

    chainer.config.train = False
    
    if args.activation == "sigmoid":
        activation = F.sigmoid
    elif args.activation == "tanh":
        activation = F.tanh
    elif args.activation == "relu":
        activation = F.relu
    else:
        print("Wrong activation function specified")
        return
    if args.rpl:
        fold = 0
        fold_models = []
        if args.master_network != "-":
            print("Loading master network")
            master_model = get_nn(args.network, args.layers, args.units, num_classes, activation, args.tdnn_ksize, args.dropout)
            master_model_cls = L.Classifier(master_model)
            chainer.serializers.load_npz(args.master_network, master_model_cls)
        else:
            master_model = None
        if args.fold_network_pattern != "-":
            while True:
                model_file = Path(args.fold_model_dir, args.fold_network_pattern.format(fold))
                if not model_file.is_file():
                    break
                print("Loading fold {} network".format(fold))
                fold_model = get_nn(args.network, args.layers, args.units, num_classes, activation, args.tdnn_ksize, args.dropout)
                fold_model_cls = L.Classifier(fold_model)
                chainer.serializers.load_npz(model_file, fold_model_cls)
                fold_models.append(fold_model)
                fold += 1
        if args.rpl_model != "-":
            rpl_model = RPL4(num_classes)
            rpl_model_cls = L.Classifier(rpl_model)
            chainer.serializers.load_npz(args.rpl_model, rpl_model_cls)
        else:
            rpl_model = None
        model = NNWithRPL(master_model, fold_models, rpl_model)
    else:
        model = get_nn(args.network, args.layers, args.units, num_classes, activation, args.tdnn_ksize, args.dropout)
        model_cls = L.Classifier(model)
        chainer.serializers.load_npz(args.model, model_cls)
    
    if args.network == "tdnn":
        splice = (sum(args.tdnn_ksize) - len(args.tdnn_ksize)) // 2
    else:
        splice = args.splice

    if args.ft is not None and args.ft != '-':
        ft = loadKaldiFeatureTransform(str(Path(args.data_dir, args.ft)))
        if is_nn_recurrent(args.network):
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

    data = np.load(str(Path(args.data_dir, args.data.format(args.test_or_dev))))
    if splice > 0:
        data = splicing(data, range(-splice, splice + 1))
    if ft is not None:
        data = applyKaldiFeatureTransform(data, ft)
    offsets = np.load(str(Path(args.offset_dir, args.offsets.format(args.test_or_dev))))
    if args.ivector_dir is not None:
        ivectors = np.load(str(Path(args.ivector_dir, args.ivectors.format(args.test_or_dev))))
        data = np.concatenate((data, ivectors), axis=1)
        
    if args.tri:
        ap = args.ap_coef * np.load(str(Path(args.recog_dir, args.ap_file)))
        per = evaluateModelTestTri(model, data, offsets, args.PIP, args.LMW, ap=ap, testOrDev=args.test_or_dev, uttlistdir=args.utt_list_dir, recogdir=args.recog_dir, GPUID=args.gpu, progress=not args.no_progress, rnn=is_nn_recurrent(args.network))
    else:
        print("Monophones not implemented")
        return
    
    print("PER: {0:.2f} %".format(per))
    
if __name__ == '__main__':
    main()
