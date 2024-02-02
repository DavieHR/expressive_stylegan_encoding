set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    files=`ls video/`
    for file in ${files[@]};
    do
        basename=`echo $file | awk -F '.' '{print $1}'`
        echo "${basename} is processing."
        CUDA_VISIBLE_DEVICES=7 python -m ExpressiveEncoding \
                               --config_path ./scripts/${exp_name} \
                               --save_path ./results/${exp_name}/${basename} \
                               --path ./video/${file}
    done
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
