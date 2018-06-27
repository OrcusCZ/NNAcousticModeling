#!/usr/bin/bash
# Simple Feed-Forward Example
./scripts/common/master_script.py --output-dir example_out --output-id example_ff --network-spec "-n ff -l 8 -u 2048 -a relu --splice 5 -d 0.2"