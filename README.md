# Introduction
This repository contains scripts for training and evaluation of neural network based acoustic models.
The scripts use Python framework Chainer which supports GPU.
These scripts were used in the following publications:
* [A Survey of Recent DNN Architectures on the TIMIT Phone Recognition Task, TSD 2018](http://arxiv.org/abs/1806.07974)
* [Recurrent DNNs and its Ensembles on the TIMIT Phone Recognition Task, SPECOM 2018](https://arxiv.org/abs/1806.07186)

The acoustic models are evaluated on the TIMIT phone recognition task.

# Repository Description
## Folder Structure
* **data** - Features, utterance offsets, targets
* **kaldi** - Baseline HMM-DNN model trained in Kaldi
* **recog** - Recognizer executable and files used only in recognizer
* **recog_src** - Recognizer source files, needed only when building the recongizer and not using precompiled executable in _recog_ folder
* **scripts** - All scripts used for training / evaluation

## Input Data
All input files specified in this README can contain the characters {}, which can be substituted with substring _train_, _dev_ or _test_ for training, development or test data. The training / evaluation scripts used in this repository also replace the characters {} by the correct substring, depending on where the data is used. This allows for simpler input data specification using fewer arguments.

The folder _data_ contains the TIMIT corpus processed and saved using several techniques.
Each feature type is saved in a separate subfolder.
All feature files have the name _data\_{}.npy_ and each subfolder also contains a feature transform in a file _final.feature_transform_.

All data files are in numpy _.npy_ format and contain concatenated training, development and test utterances. The _data_ directory contains lists of the original utterance names. The data files were generated in the same order. The offsets of each utterance or speaker are provided in files _offsets\_{}.npy_ and _offsets\_spk\_{}.npy_.

Used features are:

* **fbank40** - Filter bank values
* **fbank40norm** - Filter bank values globally normalized to zero mean and unit variance
* **mfcc** - Mel-frequency cepstral coefficients
* **mfcc_cmn_perspk** - Mel-frequency cepstral coefficients (MFCC) with cepstral mean normalization (CMN) computed per speaker
* **mfcc_cmn_perutt** - MFCC with CMN computed per utterance
* **fmllr** - feature space maximum likelihood linear regression (fMLLR) data generated using Kaldi

i-vectors are saved in these files:
* **ivectors/online/ivectors\_{}.npy** - Online i-vectors
* **ivectors/offline\_perspk/ivectors\_{}.npy** - Offline i-vectors computed per speaker
* **ivectors/offline\_perutt/ivectors\_{}.npy** - Offline i-vectors computed per utterance

The targets for training and development data are saved in _targets/targets\_{}.npy_. There are no targets for testing data.

## Scripts
The scripts are separated into several directories:
* **common** - Scripts used for training, evaluation, model specifications and master script.
* **example** - Example scripts
* **papers** - Scripts which perform the data preparation, training and evaluation with the same configuration as in our papers.
* **util** - Utility and helper scripts. They are imported from other scripts and this folder should therefore be in user's _$PYTHONPATH_.

The folder _common_ contains several scripts used in different phases of acoustic model training / evaluation:
* **generate_folds.py** - Separation of the input data into folds
* **train.py** - Training
* **predict\_folds.py** - Evaluation of the fold network outputs
* **evaluate.py** - Evaluation of the phone error rate

These scripts can be executed separatedly (which is used mostly for debugging) or the master script called **master\_script.py** can be used to execute them all in the correct order and with correct arguments.

All the scripts assume they are executed with the current working directory being the repository root directory. For example, this command can be used to execute the master script in the correct directory:
```bash
$ python ./scripts/common/master_script.py [options]
```

## Output data
All output data are saved to folder _results_. The recognizer also creates an auxiliary folder _lab_, which can be deleted after the evaluation script finishes. The output data is saved to subfolders according to output data type and also some script arguments. The resulting phone error rates (PER) are written to the stdout.

The following description uses these values, which are substituted according to the script arguments:
* **[num\_folds]** - Numer of folds used. If folds are not used, this value is equal to 0.
* **[data]** - Name of the subdirectory containing the features, for ex. _fmllr_
* **[ivectors]** - Name of the subdirectory containing the ivectors, for ex. _offline\_perspk_
* **[output\_dir]** - Name of the output directory
* **[output\_id]** - Arbitrary string describing the experiment being run. This string will be used as the name of the subdirectory containing intermediate and final results.

All output directories are:
* **Folds** - _[output\_dir]/fold\_data/[num\_folds]/[data]+[ivectors]_, if i-vectors are not used, the part _+[ivectors]_ is missing
* **Fold network outputs** - _[output\_dir]/fold\_data\_out/[num\_folds]/[output\_id]_
* **Fold network models** - _[output\_dir]/models/folds/[num\_folds]/[output\_id]_
* **Master network models** - _[output\_dir]/models/master/[num\_folds]/[output\_id]_
* **RPL network models** - _[output\_dir]/models/rpl/[num\_folds]/[output\_id]_

# Recognizer
The part of this repository is a triphone-based HMM-DNN phone recognizer.
The recognizer is written in C++.
Its source files are in the directory _recog\_src_ and CMake is used to generate project files.
Compiled recognizer executables for both Windows and GNU Linux are in the directory _recog_ together with necessary files for recognition of baseline Kaldi HMM.

# Running
## Prerequisities
* Python 3.6+
* Chainer 3.5 with CuPy
* Numpy

The scripts may work in other versions, but only these versions were tested.

### Important Notes
* The directory _scripts/util_ must be added to _$PYTHONPATH_ if executing any script in _scripts/common_ directly. Scripts in _scripts/papers_ modify the $PYTHONPATH and user can omit this change.
* All scripts should be executed with current directory being the repository root, otherwise the default paths won't be correctly found and must be set via arguments. For ex.
```bash
$ python ./scripts/common/master_script.py [options]
```
* Scripts in _scripts/papers_ must also be executed from the repository root, for ex.
```
$ ./scripts/papers/tsd2018/run.sh
```
* Use different `--output-id` for different experiments. Whenever other arguments change (mainly network specification), different output ID should be used to avoid using intermediate files meant for another network.

### Examples
#### Simple Feed-Forward Example
```
$ python ./scripts/common/master_script.py --output-dir example_out
    --output-id example_ff
    --network-spec "-n ff -l 8 -u 2048 -a relu --splice 5 -d 0.2"
```

This command trains and evaluates a feed-forward network with 8 layers and 2048 units, ReLU activation functions and dropout ratio = 0.2. The splicing size is 5 (in each direction, so in total there are 11 stacked frames in the network input). This example trains one network which is saved in the directory _example\_out/models/master/0/example\_ff_.

Note: In feed-forward case, only splicing size 5 should be used, otherwise new feature transform files should be generated.

#### Simple LSTM Example
```
$ python ./scripts/common/master_script.py --output-id example_lstm
    --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2"
```

In this example, only network architecture is different. LSTM network with 4 layers and 1024 units is used. Output time delay is 5 time steps and used dropout ratio is again 0.2.

#### LSTM with Specified Optimizer
```
$ python ./scripts/common/master_script.py --output-id example_lstm_optimizer
    --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2"
    -o adam momentumsgd
    -b 512 128
    --lr 0 1e-3 1e-4 1e-5
```
This example is the same as the last one, except it now specifies the optimizer settings.
`-o` specifies the optimizer, `-b` batch size and `--lr` the learning rate.
All of these parameters accept several arguments and training is performed in several stages. Each argument is used for one stage. If any parameter has lower number of arguments than the longest one, the last argument is automatically duplicated.

Therefore, the last command is identical with:
```
$ python ./scripts/common/master_script.py --output-id example_lstm
    --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2"
    -o adam momentumsgd momentumsgd momentumsgd
    -b 512 128 128 128
    --lr 0 1e-3 1e-4 1e-5
```
and the training stages in this example use the following settings:

1. Adam, batch size = 512
2. Momentum SGD, batch size = 128, learning rate = 1e-3
3. Momentum SGD, batch size = 128, learning rate = 1e-4
4. Momentum SGD, batch size = 128, learning rate = 1e-5

Note: Adam has no learning rate, therefore the corresponding learning rate argument (0 in this case) is not used, but must be specified anyway so the training stages are corretly set up. N-th argument of `-o`, `-b` and `--lr` belongs to N-th training stage.

#### Using Fold Networks with Regularization Post Layer (RPL)
```
$ python ./scripts/common/master_script.py --output-id example_lstm_folds
    --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2"
    -o adam momentumsgd
    -b 512 128
    --lr 0 1e-3 1e-4 1e-5
    --num-folds 5
```

This command performs several steps:

1. Separates training data into 5 folds
2. Trains master network using the full training set
3. Trains fold networks, N-th network is trained using concatenated folds except the N-th fold
4. Retrieves fold network outputs, N-th network uses N-th fold as input
5. Trains a simple network consisting only of RPL, network inputs are fold network outputs from the previous step and the targets are original training set targets
6. Creates models from all combinations of master network, fold network and RPL and evaluates them all on the test data

#### Using i-vectors
```
$ python ./scripts/common/master_script.py --output-id example_lstm_ivectors
    --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2"
    -o adam momentumsgd
    -b 512 128
    --lr 0 1e-3 1e-4 1e-5
    --ivector-dir data/ivectors/online data/ivectors/offline_perspk
```

This example is the same as _LSTM with Specified Optimizer_, except i-vectors are used now.
`--ivector-dir` parameter has 2 arguments: training and test i-vectors directories.
This parameter has no default value and if not specified, i-vectors are simply not used.

In this example, online i-vectors are used for training and offline per speaker for testing.

#### Evaluating on Both Development and Test Data
```
$ python ./scripts/common/master_script.py --output-id example_lstm_optimizer
    --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2"
    -o adam momentumsgd
    -b 512 128
    --lr 0 1e-3 1e-4 1e-5
    --eval-data dev test
```

The parameter `--eval-data` specifies, which dataset should be used for evaluation.
The accepted values are _dev_ for development data and _test_ for test data.
Both values can be specified at once and both will be evaluated in order.