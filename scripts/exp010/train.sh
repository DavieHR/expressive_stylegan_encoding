set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results


function main
{
  files=`ls /data1/Dataset/chenlong/0206/video/kanghui/`
  for file in ${files[@]};
  do
  basename=`echo $file | awk -F '.' '{print $1}'`
  echo "${basename} is processing."
  if [[ $basename == '0' || $basename == '1' ]]; 
  then
      echo 'skip ${basename} ...'
      continue
  fi
  CUDA_VISIBLE_DEVICES=5 python -m ExpressiveEncoding \
                         --config_path ./scripts/${exp_name} \
                         --save_path ./results/${exp_name}/${basename} \
                         --path /data1/Dataset/chenlong/0206/video/kanghui/${file}
  done
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`
main
