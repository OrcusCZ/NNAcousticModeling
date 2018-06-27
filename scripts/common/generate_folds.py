#!/usr/bin/env python3

# system
import argparse
from pathlib import Path
import numpy as np
# util
from kw_utils import loadBin, saveBin

def main(arg_list=None):
    parser = argparse.ArgumentParser(description='Chainer LSTM')
    parser.add_argument('-n', type=int, default=5, help='Number of folds')
    parser.add_argument('--data-dir', default='data/fmllr')
    parser.add_argument('--offset-dir', default='data')
    parser.add_argument('--target-dir', default='data/targets')
    parser.add_argument('--ivector-dir', help="Directory with i-vector files")
    parser.add_argument('--utt-list-dir', default='data', help='Directory with utterance lists')
    parser.add_argument('--fold-data-dir', default='fold_data', help='Directory with fold input data')
    parser.add_argument('--data', default='data_train.npy', help='Data')
    parser.add_argument('--offsets', default='offsets_train.npy', help='Offsets')
    parser.add_argument('--targets', default='targets_train.npy', help='Targets')
    parser.add_argument('--ivectors', default='ivectors_train.npy', help='i-vectors')
    parser.add_argument('--fold-data-pattern', default='data_{}.npy', help='Filename pattern of each fold data, {} will be replaced by fold ID')
    parser.add_argument('--fold-offset-pattern', default='offsets_{}.npy', help='Filename pattern of each fold offset, {} will be replaced by fold ID')
    parser.add_argument('--fold-target-pattern', default='targets_{}.npy', help='Filename pattern of each fold targets, {} will be replaced by fold ID')
    parser.add_argument('--fold-ivector-pattern', default='ivectors_{}.npy', help='Filename pattern of each fold i-vectors file, {} will be replaced by fold ID')
    parser.add_argument('--train-list', default='train.list', help='Filename of the training list')
    parser.add_argument('--utt-idx', default='utt_idx.npz', help="Utterance index file. If the file does not exist, it will be created, otherwise it will be used to separate utterances into folds")
    if arg_list is not None:
        args = parser.parse_args(list(map(str, arg_list)))
    else:
        args = parser.parse_args()

    # create output directories
    Path(args.fold_data_dir).mkdir(exist_ok=True, parents=True)

    # load input data
    data = np.load(str(Path(args.data_dir, args.data)))
    offsets = np.load(str(Path(args.offset_dir, args.offsets)))
    targets = np.load(str(Path(args.target_dir, args.targets)))
    if args.ivector_dir is not None:
        ivectors = np.load(str(Path(args.ivector_dir, args.ivectors)))
    else:
        ivectors = None
    train_list = [x.strip() for x in open(str(Path(args.utt_list_dir, args.train_list))).readlines()]

    # create empty lists for output data
    fold_data = [[] for k in range(args.n)]
    fold_offsets = [[0] for k in range(args.n)]
    fold_targets = [[] for k in range(args.n)]
    if ivectors is not None:
        fold_ivectors = [[] for k in range(args.n)]
    fold_counts = [0] * args.n

    utt_idx_file = Path(args.fold_data_dir, args.utt_idx)
    if args.utt_idx and utt_idx_file.is_file():
        print('Using existing utterance index file')
        # load index map from file and separate utterances into folds as specified in the index map
        utt_idx = np.load(str(utt_idx_file))
        utt_idx = list(map(lambda x: utt_idx[x], utt_idx.files))

        for k in range(args.n):
            for i in utt_idx[k]:
                beg = offsets[i]
                end = offsets[i + 1]
                fold_counts[k] += end - beg
                fold_data[k].append(data[beg:end])
                fold_offsets[k].append(fold_counts[k])
                fold_targets[k].append(targets[beg:end])
                if ivectors is not None:
                    fold_ivectors[k].append(ivectors[beg:end])
    else:
        print('No utterance index file found, creating new index map')
        # separate utterances into random folds and save their indices into index map file
        speakers = sorted(set(map(lambda x: x[:5], train_list)))
        rand_idx = np.random.random_integers(0, args.n - 1, len(speakers))
        utt_idx = [[] for k in range(args.n)]

        for i, u in enumerate(train_list):
            k = speakers.index(u[:5])
            k = rand_idx[k]
            utt_idx[k].append(i)

            beg = offsets[i]
            end = offsets[i + 1]
            fold_counts[k] += end - beg
            fold_data[k].append(data[beg:end])
            fold_offsets[k].append(fold_counts[k])
            fold_targets[k].append(targets[beg:end])
            if ivectors is not None:
                fold_ivectors[k].append(ivectors[beg:end])

        if args.utt_idx:
            for k in range(args.n):
                utt_idx[k] = np.array(utt_idx[k], dtype=np.int32)
            np.savez(str(utt_idx_file), *utt_idx)

    # concatenate all utterances in each fold to get one array per fold
    for k in range(args.n):
        fold_data[k] = np.concatenate(fold_data[k], axis=0)
        fold_offsets[k] = np.array(fold_offsets[k], np.int32)
        fold_targets[k] = np.concatenate(fold_targets[k], axis=0)
        if ivectors is not None:
            fold_ivectors[k] = np.concatenate(fold_ivectors[k], axis=0)

    # save folds to files
    for k in range(args.n):
        np.save(str(Path(args.fold_data_dir, args.fold_data_pattern.format(k))), fold_data[k])
        np.save(str(Path(args.fold_data_dir, args.fold_offset_pattern.format(k))), fold_offsets[k])
        np.save(str(Path(args.fold_data_dir, args.fold_target_pattern.format(k))), fold_targets[k])
        if ivectors is not None:
            np.save(str(Path(args.fold_data_dir, args.fold_ivector_pattern.format(k))), fold_ivectors[k])

if __name__ == '__main__':
    main()