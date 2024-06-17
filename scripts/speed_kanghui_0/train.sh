set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results


function main
{
  CUDA_VISIBLE_DEVICES=5 python -m ExpressiveEncoding \
                         --config_path ./scripts/${exp_name} \
                         --save_path  /data1/chenlong/0517/video/0522/kanghui_0/results/man3_chenl_0521 \
                         --path /data1/chenlong/0517/video/0522/kanghui_0/results/man3_chenl_0521/data/smooth
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
