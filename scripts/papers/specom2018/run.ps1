$exec="python"
$master_script="./scripts/common/master_script.py"
$output_dir="results/specom2018"

$Env:PYTHONPATH="$(pwd)\scripts\util;$Env:PYTHONPATH"

for ($i=0; $i -lt 10; $i++) {
    & $exec $master_script --output-dir $output_dir --output-id ff_$i          --num-folds 5 --network-spec "-n ff          -l 8 -u 2048 -a relu --splice 5 -d 0.2" -o momentumsgd -b 256 1024 2048 --lr 1e-2 4e-3 1e-4
    & $exec $master_script --output-dir $output_dir --output-id lstm_$i        --num-folds 5 --network-spec "-n lstm        -l 4 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    & $exec $master_script --output-dir $output_dir --output-id gru_$i         --num-folds 5 --network-spec "-n gru         -l 4 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    & $exec $master_script --output-dir $output_dir --output-id zoneoutlstm_$i --num-folds 5 --network-spec "-n zoneoutlstm -l 4 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
}