set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/f_space_detailed_training.py \
                                 --config_path ./scripts/${exp_name}/config.yaml \
                                 --save_path ./results/${exp_name} \
                                 --gpus 4 \
                                 --resume_path ./results/bdinv_002/snapshots/best.pth \
                                 --decoder_path ./results/pivot_052/snapshots/best.pth
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
