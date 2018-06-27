master_script="./scripts/common/master_script.py"
output_dir="results/tsd2018"

export PYTHONPATH="$(pwd)/scripts/util:$PYTHONPATH"

for i in {0..9}; do
    # Feed-forward
    $master_script --output-dir $output_dir --output-id ff_6_512_$i          --network-spec "-n ff -l 6 -u 512  -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_6_1024_$i         --network-spec "-n ff -l 6 -u 1024 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_6_2048_$i         --network-spec "-n ff -l 6 -u 2048 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_7_512_$i          --network-spec "-n ff -l 7 -u 512  -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_7_1024_$i         --network-spec "-n ff -l 7 -u 1024 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_7_2048_$i         --network-spec "-n ff -l 7 -u 2048 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_8_512_$i          --network-spec "-n ff -l 8 -u 512  -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_8_1024_$i         --network-spec "-n ff -l 8 -u 1024 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_8_2048_$i         --network-spec "-n ff -l 8 -u 2048 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_9_512_$i          --network-spec "-n ff -l 9 -u 512  -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_9_1024_$i         --network-spec "-n ff -l 9 -u 1024 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4
    $master_script --output-dir $output_dir --output-id ff_9_2048_$i         --network-spec "-n ff -l 9 -u 2048 -a relu    --splice 5 -d 0.2" -o momentumsgd -b 256 512 1024 2048 --lr 1e-2 4e-3 1e-3 1e-4

    # TDNN
    $master_script --output-dir $output_dir --output-id tdnn_5-5-5-5_256_$i  --network-spec "-n tdnn --tdnn-ksize 5 5 5 5 -u 256  256  256  256  -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_5-5-5-5_512_$i  --network-spec "-n tdnn --tdnn-ksize 5 5 5 5 -u 512  512  512  512  -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_5-5-5-5_1024_$i --network-spec "-n tdnn --tdnn-ksize 5 5 5 5 -u 1024 1024 1024 1024 -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_5-5-9-9_256_$i  --network-spec "-n tdnn --tdnn-ksize 5 5 9 9 -u 256  256  256  256  -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_5-5-9-9_512_$i  --network-spec "-n tdnn --tdnn-ksize 5 5 9 9 -u 512  512  512  512  -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_5-5-9-9_1024_$i --network-spec "-n tdnn --tdnn-ksize 5 5 9 9 -u 1024 1024 1024 1024 -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_9-9-9-9_256_$i  --network-spec "-n tdnn --tdnn-ksize 9 9 9 9 -u 256  256  256  256  -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_9-9-9-9_512_$i  --network-spec "-n tdnn --tdnn-ksize 9 9 9 9 -u 512  512  512  512  -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id tdnn_9-9-9-9_1024_$i --network-spec "-n tdnn --tdnn-ksize 9 9 9 9 -u 1024 1024 1024 1024 -a relu -d 0.2" -o adam momentumsgd -b 256 512 1024 2048 --lr 1e-2 1e-3 1e-4 1e-5

    # LSTM
    $master_script --output-dir $output_dir --output-id lstm_2_256_$i  --network-spec "-n lstm -l 2 -u 256  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_3_256_$i  --network-spec "-n lstm -l 3 -u 256  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_4_256_$i  --network-spec "-n lstm -l 4 -u 256  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_5_256_$i  --network-spec "-n lstm -l 5 -u 256  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_6_256_$i  --network-spec "-n lstm -l 6 -u 256  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_2_512_$i  --network-spec "-n lstm -l 2 -u 512  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_3_512_$i  --network-spec "-n lstm -l 3 -u 512  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_4_512_$i  --network-spec "-n lstm -l 4 -u 512  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_5_512_$i  --network-spec "-n lstm -l 5 -u 512  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_6_512_$i  --network-spec "-n lstm -l 6 -u 512  --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_2_1024_$i --network-spec "-n lstm -l 2 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_3_1024_$i --network-spec "-n lstm -l 3 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_4_1024_$i --network-spec "-n lstm -l 4 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_5_1024_$i --network-spec "-n lstm -l 5 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
    $master_script --output-dir $output_dir --output-id lstm_6_1024_$i --network-spec "-n lstm -l 6 -u 1024 --timedelay 5 -d 0.2" -o adam momentumsgd -b 512 128 --lr 1e-2 1e-3 1e-4 1e-5
done