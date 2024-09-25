set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    CUDA_VISIBLE_DEVICES=3,4,5,6 python tools/f_space_training.py \
                                 --config_path ./scripts/${exp_name}/config.yaml \
                                 --save_path ./results/${exp_name} \
                                 --gpus 4 \
                                 --decoder_path /data1/chenlong/0517/video/0822/YRKL6NhF/YRKL6NhF_exp_set_v4_depoly/exp/pti_ft_512/snapshots \
                                 --resume_path ./results/bdinv_003/snapshots/best.pth
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
