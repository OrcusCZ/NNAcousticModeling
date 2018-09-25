$exec="python"
$master_script="./scripts/common/master_script.py"
$output_dir="results/slsp2018"

$Env:PYTHONPATH="$(Get-Location)\scripts\util;$Env:PYTHONPATH"

function run_exp($i, $network, $data, $ivec_train, $ivec_test) {
    if ($ivec_train -and $ivec_test) {
        $ivec_arg = "--ivector-dir", $ivec_train, $ivec_test
    }
    else {
        $ivec_arg = @()
    }

    if ($network -eq "ff") {
        $layers = 8
        $units = 2048
        $splice = 5
        $td = 0
        $opt = ,"momentumsgd"
        $batch = 256, 1024, 2048
        $lr = 1e-2, 4e-3, 1e-4
    }
    else {
        $layers = 4
        $units = 1024
        $splice = 0
        $td = 5
        $opt = "adam", "momentumsgd"
        $batch = 256, 128
        $lr = 1e-2, 1e-3, 1e-4, 1e-5
    }

    $id = "${network}_${layers}_${units}_${data}_${ivec_train}_${ivec_test}_$i"
    & $exec $master_script --output-dir $output_dir --output-id $id  --network-spec "-n $network -l $layers -u $units -a relu --splice $splice --timedelay $td -d 0.2" -o @opt -b @batch --lr @lr --no-train-folds --no-predict --no-train-rpl --eval-only-master @ivec_arg
}

for ($i=0; $i -lt 10; $i++) {
    foreach ($network in @("ff", "lstm", "gru", "mgrurelu", "mgrurelur")) {
        foreach ($data in @("fmllr", "mfcc", "mfcc_cmn_spk", "mfcc_cmn_utt")) {
            run_exp $i $network $data "" ""
            run_exp $i $network $data "online" "online"
            run_exp $i $network $data "online" "offline_perspk"
            run_exp $i $network $data "online" "offline_perutt"
            run_exp $i $network $data "offline_perspk" "offline_perspk"
            run_exp $i $network $data "offline_perutt" "offline_perutt"
        }
    }
}