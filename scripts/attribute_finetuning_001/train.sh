set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
  CUDA_VISIBLE_DEVICES=6 python -m ExpressiveEncoding.attribute_train \
                         --from_path ./results/exp010/1/ \
                         --config_path ./scripts/${exp_name}/config.yaml \
                         --to_path ./results/${exp_name}
}

main
