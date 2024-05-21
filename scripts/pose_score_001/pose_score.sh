set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
  CUDA_VISIBLE_DEVICES=5 python -m ExpressiveEncoding.pose_score \
                         --training_path ./results/exp010/0/ \
                         --config_path ./scripts/pose_finetuning_007/pose.yaml \
                         --snapshots_path ./results/pose_finetuning_007 \
                         --to_path ./results/${exp_name}.json
}

main

