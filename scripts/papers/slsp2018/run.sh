master_script="./scripts/common/master_script.py"
output_dir="results/slsp2018"

export PYTHONPATH="$(pwd)/scripts/util:$PYTHONPATH"

run_exp() {
    local i=$1
    local network=$2
    local data=$3
    local ivec_train=$4
    local ivec_test=$5

    if [[ -n "$ivec_train" || -n "$ivec_test" ]]; then
        local ivec_arg=("--ivector-dir" $ivec_train $ivec_test)
    else
        local ivec_arg=()
    fi

    if [[ "$network" -eq "ff" ]]; then
        local layers=8
        local units=2048
        local splice=5
        local td=0
        local opt=("momentumsgd")
        local batch=(256 1024 2048)
        local lr=(1e-2 4e-3 1e-4)
    else
        local layers=4
        local units=1024
        local splice=0
        local td=5
        local opt=("adam" "momentumsgd")
        local batch=(256 128)
        local lr=(1e-2 1e-3 1e-4 1e-5)
    fi

    local id="${network}_${layers}_${units}_${data}_${ivec_train}_${ivec_test}_$i"
    $master_script --output-dir $output_dir --output-id $id  --network-spec "-n $network -l $layers -u $units -a relu --splice $splice --timedelay $td -d 0.2" -o "${opt[@]}" -b "${batch[@]}" --lr "${lr[@]}" --no-train-folds --no-predict --no-train-rpl --eval-only-master "${ivec_arg[@]}"
}

for ((i=0; i < 10; i++)); do
    for network in "ff" "lstm" "gru" "mgrurelu" "mgrurelur"; do
        for data in "fmllr" "mfcc" "mfcc_cmn_spk" "mfcc_cmn_utt"; do
            run_exp $i $network $data "" ""
            run_exp $i $network $data "online" "online"
            run_exp $i $network $data "online" "offline_perspk"
            run_exp $i $network $data "online" "offline_perutt"
            run_exp $i $network $data "offline_perspk" "offline_perspk"
            run_exp $i $network $data "offline_perutt" "offline_perutt"
        done
    done
done