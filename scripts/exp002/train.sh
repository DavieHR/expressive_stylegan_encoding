set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results


function main
{
    DEBUG=1 \
    CUDA_VISIBLE_DEVICES=7 python -m ExpressiveEncoding \
                           --config_path ./scripts/${exp_name} \
                           --save_path ./results/${exp_name}
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
