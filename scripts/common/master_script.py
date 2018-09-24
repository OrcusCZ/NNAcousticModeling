#!/usr/bin/env python3

# system
import argparse
from pathlib import Path
# common
from train import main as train_main
from evaluate import main as evaluate_main
from generate_folds import main as generate_folds_main
from predict_folds import main as predict_folds_main
# util
from orcus_util import str2bool

def main():
    parser = argparse.ArgumentParser(description='RPL experiments master script')
    parser.add_argument('--num-folds', type=int, default=0, help="Number of folds used, 0 to disable folds")
    parser.add_argument('--data-dir', default='data/fmllr')
    parser.add_argument('--offset-dir', default='data')
    parser.add_argument('--target-dir', default='data/targets')
    parser.add_argument('--ivector-dir', nargs=2, help="Directory with i-vector files. Expects 2 arguments: train and test i-vector directories")
    parser.add_argument('--utt-list-dir', default='data', help='Directory with utterance lists')
    parser.add_argument('--recog-dir', default='recog')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--data', default='data_{}.npy')
    parser.add_argument('--offsets', default='offsets_{}.npy')
    parser.add_argument('--targets', default='targets_{}.npy')
    parser.add_argument('--ivectors', default='ivectors_{}.npy')
    parser.add_argument('--ft', default='final.feature_transform')
    parser.add_argument('--output-id', default='tmp')
    parser.add_argument('--network-spec', default='-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2')
    parser.add_argument('--rpl-train-setup', default='-b 1024 --epoch 20 -o adam --lr 1e-3')
    parser.add_argument('--epoch', '-e', type=int, nargs='+', default=[20], help='Number of sweeps over the dataset to train')
    parser.add_argument('--optimizer', '-o', nargs='+', default=['adam', 'momentumsgd'])
    parser.add_argument('--batch-size', '-b', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4, 1e-5], help='Learning rate')
    parser.add_argument('--early-stopping', type=str2bool, nargs='+', default=[True], help="True if early stopping should be enabled")
    parser.add_argument('--fold-data-dir', help='Directory with fold input data')
    parser.add_argument('--fold-output-dir', help='Directory with predicted fold output')
    parser.add_argument('--fold-model-dir', help='Directory with output fold model')
    parser.add_argument('--fold-output-dev', default='data_dev.npy', help='Output file with predicted development data values')
    parser.add_argument('--fold-data-pattern', default='data_{}.npy', help='Filename pattern of each fold data, {1} will be replaced by fold ID')
    parser.add_argument('--fold-offset-pattern', default='offsets_{}.npy', help='Filename pattern of each fold offset')
    parser.add_argument('--fold-target-pattern', default='targets_{}.npy', help='Filename pattern of each fold targets')
    parser.add_argument('--fold-ivector-pattern', default='ivectors_{}.npy', help='Filename pattern of each fold i-vectors file, {} will be replaced by fold ID')
    parser.add_argument('--fold-output-pattern', default='data_{}.npy', help='Filename pattern of each fold network output')
    parser.add_argument('--fold-network-pattern', default='fold_{}.npz', help='Filename pattern of each fold network')
    parser.add_argument('--master-dir')
    parser.add_argument('--rpl-dir')
    parser.add_argument('--PIP', type=float, default=20)
    parser.add_argument('--LMW', type=float, default=1)
    parser.add_argument('--gen-folds', action='store_true')
    parser.add_argument('--no-train-master', action='store_true')
    parser.add_argument('--no-train-folds', action='store_true')
    parser.add_argument('--no-predict', action='store_true')
    parser.add_argument('--no-train-rpl', action='store_true')
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--eval-only-master', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--eval-data', nargs='+', default=['test'])
    args = parser.parse_args()

    # set default arguments depending on other arguments
    if args.fold_data_dir is None:
        if args.ivector_dir:
            args.fold_data_dir = "{}/fold_data/{}/{}+{}".format(args.output_dir, args.num_folds, Path(args.data_dir).name, Path(args.ivector_dir[0]).name)
        else:
            args.fold_data_dir = "{}/fold_data/{}/{}".format(args.output_dir, args.num_folds, Path(args.data_dir).name)
    if args.fold_output_dir is None:
        args.fold_output_dir = "{}/fold_data_out/{}/{}".format(args.output_dir, args.num_folds, args.output_id)
    if args.fold_model_dir is None:
        args.fold_model_dir = "{}/models/folds/{}/{}".format(args.output_dir, args.num_folds, args.output_id)
    if args.master_dir is None:
        args.master_dir = "{}/models/master/{}/{}".format(args.output_dir, args.num_folds, args.output_id)
    if args.rpl_dir is None:
        args.rpl_dir = "{}/models/rpl/{}/{}".format(args.output_dir, args.num_folds, args.output_id)

    # generate folds
    if args.num_folds > 0 and args.gen_folds:
        print("==== Generating folds")
        cmd = ["-n", args.num_folds,
            "--data-dir", args.data_dir,
            "--offset-dir", args.offset_dir,
            "--target-dir", args.target_dir,
            "--fold-data-dir", args.fold_data_dir,
            "--utt-list-dir", args.utt_list_dir,
            "--train-list", "train.list",
            "--data", args.data.format("train"),
            "--offsets", args.offsets.format("train"),
            "--targets", args.targets.format("train"),
            "--fold-data-pattern", args.fold_data_pattern,
            "--fold-offset-pattern", args.fold_offset_pattern,
            "--fold-target-pattern", args.fold_target_pattern]
        if args.ivector_dir:
            cmd += ['--ivector-dir', args.ivector_dir[0],
                "--fold-ivector-pattern", args.fold_ivector_pattern]
        generate_folds_main(cmd)
    else:
        print("==== Skipping fold data generation")

    # train master network
    if not args.no_train_master:
        print("==== Training master network")
        cmd = ["--tri",
            "--noplot",
            "-b"]
        cmd += args.batch_size
        cmd += ["--epoch"]
        cmd += args.epoch
        cmd += ["-o"]
        cmd += args.optimizer
        cmd += ["--lr"]
        cmd += args.lr
        cmd += ["--early-stopping"]
        cmd += args.early_stopping
        cmd += ["--data-dir", args.data_dir,
            "--offset-dir", args.offset_dir,
            "--target-dir", args.target_dir,
            "--data", args.data,
            "--offsets", args.offsets,
            "--targets", args.targets,
            "--ivectors", args.ivectors,
            "--ft", args.ft,
            "--out", args.master_dir]
        cmd += args.network_spec.split()
        if args.ivector_dir:
            cmd += ['--ivector-dir', args.ivector_dir[0]]
        if args.no_progress:
            cmd += ['--no-progress']
        train_main(cmd)
    else:
        print("==== Skipping training master network")

    # train folds
    if args.num_folds > 0 and not args.no_train_folds:
        for fold in range(args.num_folds):
            print("==== Training fold {}".format(fold))
            cmd = ["--tri",
                "--noplot",
                "-b"]
            cmd += args.batch_size
            cmd += ["--epoch"]
            cmd += args.epoch
            cmd += ["-o"]
            cmd += args.optimizer
            cmd += ["--lr"]
            cmd += args.lr
            cmd += ["--early-stopping"]
            cmd += args.early_stopping
            cmd += ["--data-dir", args.data_dir,
                "--offset-dir", args.offset_dir,
                "--target-dir", args.target_dir,
                "--data", args.data,
                "--offsets", args.offsets,
                "--targets", args.targets,
                "--ivectors", args.ivectors,
                "--ft", args.ft,
                "--train-fold", fold,
                "--fold-data-dir", args.fold_data_dir,
                "--fold-model-dir", args.fold_model_dir,
                "--fold-data-pattern", args.fold_data_pattern,
                "--fold-offset-pattern", args.fold_offset_pattern,
                "--fold-target-pattern", args.fold_target_pattern,
                "--out", "result_fold_tmp"]
            cmd += args.network_spec.split()
            if args.ivector_dir:
                cmd += ['--ivector-dir', args.ivector_dir[0],
                    "--fold-ivector-pattern", args.fold_ivector_pattern]
            if args.no_progress:
                cmd += ['--no-progress']
            train_main(cmd)
    else:
        print("==== Skipping training folds")

    # get fold outputs
    if args.num_folds > 0 and not args.no_predict:
        print("==== Predicting training data")
        cmd = ["--tri",
            "--ft", args.ft,
            "--fold-data-dir", args.fold_data_dir, 
            "--fold-output-dir", args.fold_output_dir,
            "--fold-model-dir", args.fold_model_dir,
            "--fold-data-pattern", args.fold_data_pattern,
            "--fold-offset-pattern", args.fold_offset_pattern,
            "--fold-network-pattern", args.fold_network_pattern,
            "--fold-output-pattern", args.fold_output_pattern]
        cmd += args.network_spec.split()
        if args.ivector_dir:
            cmd += ['--ivector-dir', args.ivector_dir[0],
                "--fold-ivector-pattern", args.fold_ivector_pattern]
        if args.no_progress:
            cmd += ['--no-progress']
        predict_folds_main(cmd)

        print("==== Predicting development data")
        cmd = ["--tri",
            "--ft", args.ft,
            "--data-dir", args.data_dir,
            "--offset-dir", args.offset_dir,
            "--data", args.data,
            "--offsets", args.offsets,
            "--fold-output-dir", args.fold_output_dir,
            "--fold-model-dir", args.fold_model_dir,
            "--fold-network-pattern", args.fold_network_pattern,
            "--fold-output-dev", args.fold_output_dev]
        cmd += args.network_spec.split()
        if args.ivector_dir:
            cmd += ['--ivector-dir', args.ivector_dir[0],
                "--fold-ivector-pattern", args.fold_ivector_pattern]
        if args.no_progress:
            cmd += ['--no-progress']
        predict_folds_main(cmd)
    else:
        print("==== Skipping predicting training and development data")

    # train RPL layer
    if args.num_folds > 0 and not args.no_train_rpl:
        print("==== Training RPL layer")
        cmd = ["--train-rpl",
            "--tri",
            "--data-dir", args.fold_output_dir,
            "--target-dir", args.target_dir,
            "--data", args.fold_output_dev,
            "--targets", args.targets,
            "--fold-data-dir", args.fold_data_dir, 
            "--fold-output-dir", args.fold_output_dir,
            "--fold-output-pattern", args.fold_output_pattern,
            "--fold-target-pattern", args.fold_target_pattern,
            "--out", args.rpl_dir]
        cmd += args.rpl_train_setup.split()
        if args.no_progress:
            cmd += ['--no-progress']
        train_main(cmd)
    else:
        print("==== Skipping training RPL layer")

    # evaluate
    if not args.no_eval:
        for eval_data in args.eval_data:
            print("==== Evaluating {} data".format(eval_data))
            for eval_folds in [False, True]:
                for eval_master in [False, True]:
                    for eval_rpl in [False, True]:
                        if (args.num_folds == 0 or args.eval_only_master) and (eval_folds or not eval_master or eval_rpl):
                            continue
                        if eval_folds or eval_master:
                            print("==== Evaluating {}folds {}master {}rpl".format("+" if eval_folds else "-", "+" if eval_master else "-", "+" if eval_rpl else "-"))
                            cmd = ["--tri",
                                "--data-dir", args.data_dir,
                                "--offset-dir", args.offset_dir,
                                "--utt-list-dir", args.utt_list_dir,
                                "--data", args.data,
                                "--offsets", args.offsets,
                                "--ivectors", args.ivectors,
                                "--ft", args.ft,
                                "--recog-dir", args.recog_dir,
                                "--rpl",
                                "--rpl-model", "{}/model".format(args.rpl_dir) if eval_rpl else "-",
                                "--master-network", "{}/model".format(args.master_dir) if eval_master else "-",
                                "--PIP", args.PIP,
                                "--LMW", args.LMW,
                                "--fold-model-dir", args.fold_model_dir,
                                "--fold-network-pattern", args.fold_network_pattern if eval_folds else "-",
                                "--test-or-dev", eval_data]
                            cmd += args.network_spec.split()
                            if args.ivector_dir:
                                cmd += ['--ivector-dir', args.ivector_dir[1]]
                            if args.no_progress:
                                cmd += ['--no-progress']
                            evaluate_main(cmd)
    else:
        print("==== Skipping evaluation")

if __name__ == '__main__':
    main()
