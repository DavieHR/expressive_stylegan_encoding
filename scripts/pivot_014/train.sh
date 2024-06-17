set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    CUDA_VISIBLE_DEVICES=7 python tools/pivot_training.py \
                           --config_path ./scripts/${exp_name}/config.yaml \
                           --save_path ./results/${exp_name} \
                           --resume_path ./results/pivot_014/snapshots/4.pth
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
