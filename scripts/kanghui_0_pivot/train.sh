set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    CUDA_VISIBLE_DEVICES=6 python tools/pivot_training.py \
                           --config_path ./scripts/${exp_name}/config.yaml \
                           --save_path /data1/chenlong/0517/video/0522/kanghui_0/results/man3_chenl_0521/${exp_name}_123 \
#                           --resume_path ./results/exp010/0/pti/snapshots/100.pth
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
