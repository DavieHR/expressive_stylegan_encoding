set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

exp_name=$1
echo $exp_name

function main
{
    python ExpressiveEncoding/pivot_training_v3_stage1.py \
                           --config_path sh_scripts/pivot_v3_stage1/config.yaml \
                           --save_path /data1/chenlong/0517/video/0612/man3_test_v3/v3_pti_512 \
                           --gt_path /data1/chenlong/0517/video/0612/smooth/1 \
                           --latent_path /data1/chenlong/0517/video/0612/man3_test_v3/s_pti
}

_timestamp=`date +%Y%m%d%H`
main
