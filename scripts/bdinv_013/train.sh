set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

exp_dir=/data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp
decoder_path=${exp_dir}/pti_ft_512/facial_snapshots_ft_512
mkdir -p results

function main
{
    CUDA_VISIBLE_DEVICES=3,4,5,6 python tools/f_space_training.py \
                                 --config_path ./scripts/${exp_name}/config.yaml \
                                 --save_path ./results/${exp_name} \
                                 --gpus 4 \
                                 --decoder_path ${decoder_path} 
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
export MASTER_PORT=25558
main
