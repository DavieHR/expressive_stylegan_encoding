set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{ 
  CUDA_VISIBLE_DEVICES=5 python -m ExpressiveEncoding.pose_train \
                         --training_path ./results/exp010/1/ \
                         --config_path ./scripts/${exp_name} \
                         --snapshots_path ./results/exp010/1/
}

main
