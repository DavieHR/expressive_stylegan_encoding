set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{ 
  DEBUG=True \
  CUDA_VISIBLE_DEVICES=6 python -m ExpressiveEncoding.pose_train \
                         --training_path ./results/kanghui \
                         --config_path ./scripts/${exp_name} \
                         --resume_path ./results/pose_finetuning_022/param \
                         --snapshots_path ./results/${exp_name} \
                         --option_config_path ./scripts/${exp_name}/config.yaml
}

main
