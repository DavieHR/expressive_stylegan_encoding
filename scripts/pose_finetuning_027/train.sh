set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
  CUDA_VISIBLE_DEVICES=6 python -m ExpressiveEncoding.pose_train \
                         --training_path ./results/exp010/0/ \
                         --config_path ./scripts/${exp_name} \
                         --resume_path ./results/pose_finetuning_023/param \
                         --snapshots_path ./results/${exp_name} \
                         --option_config_path ./scripts/${exp_name}/config.yaml \
                         --decoder_path /data1/chenlong/0517/video/0522/kanghui_0/results/kanghui_w_speed/pti/w_snapshots/26.pth
}

main
