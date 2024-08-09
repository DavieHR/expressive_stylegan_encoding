set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
username=`whoami`

#exp_name=$1
echo $exp_name

function main
{
  CUDA_VISIBLE_DEVICES=2 python -m ExpressiveEncoding \
                         --config_path sh_scripts/train_v3 \
                         --save_path /data1/chenlong/0517/video/0612/man3_test_v3 \
                         --path /data1/chenlong/0517/video/0612/smooth/1 \
                         --pipeline train_v3
}

_timestamp=`date +%Y%m%d%H`
main
