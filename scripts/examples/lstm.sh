#!/usr/bin/bash
# Simple LSTM Example
./scripts/common/master_script.py --output-dir example_out --output-id example_lstm --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2"