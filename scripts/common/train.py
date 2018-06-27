#!/usr/bin/env python3

# system
import os
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions, StandardUpdater
from chainer.iterators import SerialIterator
# common
from chainer_networks import get_nn, is_nn_recurrent
from RPL import RPL0, RPL, RPL2, RPL3, RPL4
# util
from orcus_util import index_padded, apply_time_delay
from orcus_chainer_util import model_saver, SequenceShuffleIterator, BPTTUpdater
from kw_utils import loadBin, splicing
from kw_nn_utils import loadKaldiFeatureTransform, applyKaldiFeatureTransform
from chainer_kw_utils import EarlyStoppingTrigger

def main(arg_list=None):
    parser = argparse.ArgumentParser(description='Chainer LSTM')
    parser.add_argument('--epoch', '-e', type=int, nargs='+', default=[20], help='Number of sweeps over the dataset to train')
    parser.add_argument('--optimizer', '-o', nargs='+', default=['momentumsgd'], help='Optimizer (sgd, momentumsgd, adam)')
    parser.add_argument('--batchsize', '-b', type=int, nargs='+', default=[128], help='Number of training points in each mini-batch')
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4, 1e-5], help='Learning rate')
    parser.add_argument('--network', '-n', default='ff', help='Neural network type, either "ff", "tdnn", "lstm", "zoneoutlstm", "peepholelstm" or "gru". Setting any recurrent network implies "--shuffle-sequences"')
    parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', default='result', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--units', '-u', type=int, nargs='+', default=[1024], help='Number of units')
    parser.add_argument('--layers', '-l', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--activation', '-a', default='relu', help='FF activation function (sigmoid, tanh or relu)')
    parser.add_argument('--tdnn-ksize', type=int, nargs='+', default=[5], help='TDNN kernel size')
    parser.add_argument('--bproplen', type=int, default=20, help='Backpropagation length')
    parser.add_argument('--timedelay', type=int, default=0, help='Delay target values by this many time steps')
    parser.add_argument('--noplot', dest='plot', action='store_false', help='Disable PlotReport extension')
    parser.add_argument('--splice', type=int, default=0, help='Splicing size')
    parser.add_argument('--dropout', '-d', type=float, nargs='+', default=[0], help='Dropout rate (0 to disable). In case of Zoneout LSTM, this parameter has 2 arguments: c_ratio h_ratio')
    parser.add_argument('--ft', default='final.feature_transform', help='Kaldi feature transform file')
    parser.add_argument('--tri', action='store_true', help='Use triphones')
    parser.add_argument('--shuffle-sequences', action='store_true', help='True if sequences should be shuffled as a whole, otherwise all frames will be shuffled independent of each other')
    parser.add_argument('--data-dir', default='data/fmllr', help='Data directory, this will be prepended to data files and feature transform')
    parser.add_argument('--offset-dir', default='data', help='Data directory, this will be prepended to offset files')
    parser.add_argument('--target-dir', default='data/targets', help='Data directory, this will be prepended to target files')
    parser.add_argument('--ivector-dir', help='Data directory, this will be prepended to ivector files')
    parser.add_argument('--data', default='data_{}.npy', help='Training data')
    parser.add_argument('--offsets', default='offsets_{}.npy', help='Training offsets')
    parser.add_argument('--targets', default='targets_{}.npy', help='Training targets')
    parser.add_argument('--ivectors', default='ivectors_{}.npy', help='Training ivectors')
    parser.add_argument('--no-validation', dest='use_validation', action='store_false', help='Do not evaluate validation data while training')
    parser.add_argument('--train-fold', type=int, help='Train fold network with this ID')
    parser.add_argument('--train-rpl', action='store_true', help='Train RPL layer')
    parser.add_argument('--rpl-model', default="result_rpl/model", help='RPL layer model')
    parser.add_argument('--fold-data-dir', default="fold_data", help='Directory with fold input data')
    parser.add_argument('--fold-output-dir', default="fold_data_out", help='Directory with predicted fold output')
    parser.add_argument('--fold-model-dir', default="fold_models", help='Directory with output fold model')
    parser.add_argument('--fold-data-pattern', default='data_{0}.npy', help='Filename pattern of each fold data, {0} will be replaced by fold ID')
    parser.add_argument('--fold-offset-pattern', default='offsets_{0}.npy', help='Filename pattern of each fold offset')
    parser.add_argument('--fold-target-pattern', default='targets_{0}.npy', help='Filename pattern of each fold targets')
    parser.add_argument('--fold-ivector-pattern', default='ivectors_{}.npy', help='Filename pattern of each fold i-vectors file, {} will be replaced by fold ID')
    parser.add_argument('--fold-output-pattern', default='data_{0}.npy', help='Filename pattern of each fold network output')
    parser.add_argument('--fold-network-pattern', default='fold_{0}.npz', help='Filename pattern of each fold network')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    if arg_list is not None:
        args = parser.parse_args(list(map(str, arg_list)))
    else:
        args = parser.parse_args()
    
    # set options implied by other options
    if is_nn_recurrent(args.network):
        args.shuffle_sequences = True
    
    # create output directories
    Path(args.out).mkdir(exist_ok=True, parents=True)
    if args.train_fold is not None:
        file_out = Path(args.fold_model_dir, args.fold_network_pattern.format(args.train_fold))
        Path(file_out.parent).mkdir(exist_ok=True, parents=True)
    
    # print arguments to the file
    with open(args.out + "/args.txt", "w") as f:
        for attr in dir(args):
            if not attr.startswith('_'):
                f.write('# {}: {}\n'.format(attr, getattr(args, attr)))
        f.write(' '.join(map(lambda x: "'" + x + "'" if ' ' in x else x, sys.argv)) + '\n')
        
    # print arguments to stdout
    for attr in dir(args):
        if not attr.startswith('_'):
            print('# {}: {}'.format(attr, getattr(args, attr)))
    print('')
    
    # input feature vector length
    num_classes = 1909 if args.tri else 39
    
    # create model
    if args.train_rpl:
        model = RPL4(num_classes)
        model_cls = L.Classifier(model)
    else:
        if args.activation == "sigmoid":
            activation = F.sigmoid
        elif args.activation == "tanh":
            activation = F.tanh
        elif args.activation == "relu":
            activation = F.relu
        else:
            print("Wrong activation function specified")
            return
        model = get_nn(args.network, args.layers, args.units, num_classes, activation, args.tdnn_ksize, args.dropout)
    
        # classifier reports softmax cross entropy loss and accuracy at every
        # iteration, which will be used by the PrintReport extension below.
        model_cls = L.Classifier(model)
    if args.gpu >= 0:
        # make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model_cls.to_gpu()  # copy the model to the GPU

    offsets = offsets_dev = None

    if args.train_rpl:
        # load training data
        fold = 0
        x = []
        y = []
        
        while True:
            x_file = Path(args.fold_output_dir, args.fold_output_pattern.format(fold))
            y_file = Path(args.fold_data_dir, args.fold_target_pattern.format(fold))
            if not x_file.is_file() or not y_file.is_file():
                break
            print("Loading fold {} data".format(fold))
            x_ = np.load(str(x_file))
            y_ = np.load(str(y_file))
            x.append(x_)
            y.append(y_)
            fold += 1
            
        if fold == 0:
            print("Error: No fold data found")
            return

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        
        if args.use_validation: #TODO: use args.data instead of args.dev_data
            x_dev = np.load(str(Path(args.data_dir, args.data.format("dev"))))
            # offsets_dev = loadBin(str(Path(args.datadir, args.dev_offsets)), np.int32)
            y_dev = np.load(str(Path(args.target_dir, args.targets.format("dev"))))
    else:
        # load training data
        ivectors = None
        ivectors_dev = None
        if args.train_fold is not None:
            x = []
            offsets = [0]
            y = []
            ivectors = []
            num = 0
            fold = 0
            while True:
                if fold != args.train_fold:
                    x_file = Path(args.fold_data_dir, args.fold_data_pattern.format(fold))
                    if not x_file.is_file():
                        break
                    offsets_file = Path(args.fold_data_dir, args.fold_offset_pattern.format(fold))
                    y_file = Path(args.fold_data_dir, args.fold_target_pattern.format(fold))
                    if args.ivector_dir is not None:
                        ivectors_file = Path(args.fold_data_dir, args.fold_ivector_pattern.format(fold))
                        if not ivectors_file.is_file():
                            print("Error: missing ivectors for fold data {}".format(fold))
                            return
                    
                    print("Loading fold {} data".format(fold))
                    x_fold = np.load(str(x_file))
                    x.append(x_fold)
                    if is_nn_recurrent(args.network):
                        offsets_fold = np.load(str(offsets_file))
                        offsets.extend(offsets_fold[1:] + num)
                    y_fold = np.load(str(y_file))
                    y.append(y_fold)
                    if args.ivector_dir is not None:
                        ivectors_fold = np.load(str(ivectors_file))
                        ivectors.append(ivectors_fold)
                    num += x_fold.shape[0]
                fold += 1
                    
            if len(x) == 0:
                print("Error: No fold data found")
                return
                
            x = np.concatenate(x, axis=0)
            if is_nn_recurrent(args.network):
                offsets = np.array(offsets, dtype=np.int32)
            y = np.concatenate(y, axis=0)
            if args.ivector_dir is not None:
                ivectors = np.concatenate(ivectors, axis=0)
        else:
            x = np.load(str(Path(args.data_dir, args.data.format("train"))))
            if is_nn_recurrent(args.network):
                offsets = np.load(str(Path(args.offset_dir, args.offsets.format("train"))))
            y = np.load(str(Path(args.target_dir, args.targets.format("train"))))
            if args.ivector_dir is not None:
                ivectors = np.load(str(Path(args.ivector_dir, args.ivectors.format("train"))))
        
        if args.use_validation:
            x_dev = np.load(str(Path(args.data_dir, args.data.format("dev"))))
            if is_nn_recurrent(args.network):
                offsets_dev = np.load(str(Path(args.offset_dir, args.offsets.format("dev"))))
            y_dev = np.load(str(Path(args.target_dir, args.targets.format("dev"))))
            if args.ivector_dir is not None:
                ivectors_dev = np.load(str(Path(args.ivector_dir, args.ivectors.format("dev"))))
    
        # apply splicing
        if args.network == "tdnn":
            splice = (sum(args.tdnn_ksize) - len(args.tdnn_ksize)) // 2
        else:
            splice = args.splice
        if splice > 0:
            x = splicing(x, range(-splice, splice + 1))
            x_dev = splicing(x_dev, range(-splice, splice + 1))
        
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
            # apply feature transform
            x = applyKaldiFeatureTransform(x, ft)
            if args.use_validation:
                x_dev = applyKaldiFeatureTransform(x_dev, ft)

        if ivectors is not None:
            x = np.concatenate((x, ivectors), axis=1)
        if ivectors_dev is not None:
            x_dev = np.concatenate((x_dev, ivectors_dev), axis=1)

        # shift the input dataset according to time delay
        if is_nn_recurrent(args.network) and args.timedelay != 0:
            x, y, offsets = apply_time_delay(x, y, offsets, args.timedelay)
            if args.use_validation:
                x_dev, y_dev, offsets_dev = apply_time_delay(x_dev, y_dev, offsets_dev, args.timedelay)
        
    # create chainer datasets
    train_dataset = chainer.datasets.TupleDataset(x, y)
    if args.use_validation:
        dev_dataset = chainer.datasets.TupleDataset(x_dev, y_dev)
        
    # prepare train stages
    train_stages_len = max(len(args.batchsize), len(args.lr))
    train_stages = [{
        'epoch': index_padded(args.epoch, i),
        'opt': index_padded(args.optimizer, i),
        'bs': index_padded(args.batchsize, i),
        'lr': index_padded(args.lr, i)}
        for i in range(train_stages_len)]
        
    for i, ts in enumerate(train_stages):
        if ts['opt'] == 'adam': # learning rate not used, don't print it
            print("=== Training stage {}: epoch = {}, batchsize = {}, optimizer = {}".format(i, ts['epoch'], ts['bs'], ts['opt']))
        else:
            print("=== Training stage {}: epoch = {}, batchsize = {}, optimizer = {}, learning rate = {}".format(i, ts['epoch'], ts['bs'], ts['opt'], ts['lr']))

        # reset state to allow training with different batch size in each stage
        if not args.train_rpl and is_nn_recurrent(args.network):
            model.reset_state()
            
        # setup an optimizer
        if ts['opt'] == "sgd":
            optimizer = chainer.optimizers.SGD(lr=ts['lr'])
        elif ts['opt'] == "momentumsgd":
            optimizer = chainer.optimizers.MomentumSGD(lr=ts['lr'])
        elif ts['opt'] == "adam":
            optimizer = chainer.optimizers.Adam()
        else:
            print("Wrong optimizer specified: {}".format(ts['opt']))
            exit(1)
        optimizer.setup(model_cls)

        if args.shuffle_sequences:
            train_iter = SequenceShuffleIterator(train_dataset, offsets, ts['bs'])
            if args.use_validation:
                dev_iter = SequenceShuffleIterator(dev_dataset, None, ts['bs'], repeat=False, shuffle=False)
        else:
            train_iter = SerialIterator(train_dataset, ts['bs'])
            if args.use_validation:
                dev_iter = SerialIterator(dev_dataset, ts['bs'], repeat=False, shuffle=False)

        # set up a trainer
        if is_nn_recurrent(args.network):
            updater = BPTTUpdater(train_iter, optimizer, args.bproplen, device=args.gpu)
        else:
            updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
        if args.use_validation:
            stop_trigger = EarlyStoppingTrigger(ts['epoch'], key='validation/main/loss', eps=-0.001)
        else:
            stop_trigger = (ts['epoch'], 'epoch')
        trainer = training.Trainer(updater, stop_trigger, out="{}/{}".format(args.out, i))
        
        trainer.extend(model_saver)

        # evaluate the model with the development dataset for each epoch
        if args.use_validation:
            trainer.extend(extensions.Evaluator(dev_iter, model_cls, device=args.gpu))

        # dump a computational graph from 'loss' variable at the first iteration
        # the "main" refers to the target link of the "main" optimizer.
        trainer.extend(extensions.dump_graph('main/loss'))

        # take a snapshot for each specified epoch
        frequency = ts['epoch'] if args.frequency == -1 else max(1, args.frequency)
        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

        # write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())

        # save two plot images to the result dir
        if args.plot and extensions.PlotReport.available():
            plot_vars_loss = ['main/loss']
            plot_vars_acc = ['main/accuracy']
            if args.use_validation:
                plot_vars_loss.append('validation/main/loss')
                plot_vars_acc.append('validation/main/accuracy')
            trainer.extend(extensions.PlotReport(plot_vars_loss, 'epoch', file_name='loss.png'))
            trainer.extend(extensions.PlotReport(plot_vars_acc, 'epoch', file_name='accuracy.png'))

        # print selected entries of the log to stdout
        # here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
        if args.use_validation:
            print_report_vars = ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']
        else:
            print_report_vars = ['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']
        trainer.extend(extensions.PrintReport(print_report_vars))

        # print a progress bar to stdout
        # trainer.extend(extensions.ProgressBar())

        if args.resume:
            # Resume from a snapshot
            chainer.serializers.load_npz(args.resume, trainer)

        # Run the training
        trainer.run()
        
        # load the last model if the max epoch was not reached (that means early stopping trigger stopped training
        # because the validation loss increased)
        if updater.epoch_detail < ts['epoch']:
            chainer.serializers.load_npz("{}/{}/model_tmp".format(args.out, i), model_cls)
        
        # remove temporary model from this training stage
        os.remove("{}/{}/model_tmp".format(args.out, i))
        
    # save the final model
    chainer.serializers.save_npz("{}/model".format(args.out), model_cls)
    if args.train_fold is not None:
        chainer.serializers.save_npz(str(Path(args.fold_model_dir, args.fold_network_pattern.format(args.train_fold))), model_cls)

if __name__ == '__main__':
    main()
