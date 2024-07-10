set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
  CUDA_VISIBLE_DEVICES=2 python -m ExpressiveEncoding.pose_train \
                         --training_path /data1/chenlong/online_model_set/exp_ori/Yd7ehnqJ/results/ \
                         --config_path ./scripts/${exp_name} \
                         --snapshots_path ./results/${exp_name} \
                         --option_config_path ./scripts/${exp_name}/config.yaml \
                         --resume_path /data1/chenlong/online_model_set/exp_ori/Yd7ehnqJ/results/expressive

}

main
