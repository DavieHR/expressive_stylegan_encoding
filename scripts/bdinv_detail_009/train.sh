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
    python tools/f_space_detailed_training.py \
                                 --config_path ./scripts/${exp_name}/config.yaml \
                                 --save_path ./results/${exp_name} \
                                 --gpus 4 \
                                 --decoder_path ${decoder_path} \
                                 --resume_path ./results/bdinv_detail_005/snapshots \
                                 --encoder_resume_path /data1/chenlong/0517/video/0822/eR6CMmTu_2/eR6CMmTu_2_exp_set_v4_depoly/exp/f_space_20/encoder/snapshots/19.pth 

}

_timestamp=`date +%Y%m%d%H`
export MASTER_PORT=15582
main
