set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{ 
  CUDA_VISIBLE_DEVICES=3 python -m ExpressiveEncoding.pose_train \
                         --training_path ./results/exp010/0/ \
                         --config_path ./scripts/${exp_name} \
                         --snapshots_path ./results/${exp_name} \
                         --resume_path ./results/pose_finetuning_011/param \
                         --decoder_path ./results/pivot_004/snapshots/243.pth
}

main
