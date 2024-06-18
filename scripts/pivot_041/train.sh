set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    DEBUG=True \
    CUDA_VISIBLE_DEVICES=5 python tools/pivot_training.py \
                           --config_path ./scripts/${exp_name}/config.yaml \
                           --save_path ./results/${exp_name} \
                           --resume_path /data1/chenlong/0517/video/0522/2/results/man3_chenl_0521/man3_pivot_512_10000/snapshots/200.pth
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
