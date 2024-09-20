set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
username=`whoami`

config_path=sh_scripts/train
save_path=/data1/chenlong/0517/video/0612/man3_test_100_0829_1
face_path=/data1/chenlong/0517/video/0612/smooth/100
mode=PTI
gpu_numbers=4
function main
{
  # get w_latent_code
  CUDA_VISIBLE_DEVICES=0 python tools/w_pivot_training_init.py \
                         --config_path ${config_path} \
                         --save_path ${save_path} \
                         --path ${face_path}
  # train_w_pti
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=${gpu_numbers} --master_port 15575 tools/w_pivot_training_multi_gpus.py ${gpu_numbers} ${config_path} ${save_path} ${face_path} ${mode}

#   get_val_pose
#  CUDA_VISIBLE_DEVICES=0 python tools/gen_pose_imgs.py \
#                         --w_pti_ckpt_path ${save_path}/pti/w_snapshots\
#                         --save_path ${save_path}/pose_val_imgs \
#                         --pose_path ${save_path}/pose \
#                         --gt_path ${face_path}
  # get_pose_and_facial
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_pose_facial_multi_gpus.py \
                         --config_path ${config_path} \
                         --save_path ${save_path} \
                         --path ${face_path} \
                         --gpu_numbers ${gpu_numbers}

  # train_s_pti
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=${gpu_numbers} --master_port 15575 tools/pivot_training_multi_gpus.py ${gpu_numbers} ${config_path} ${save_path} ${face_path} ${mode}
}
_timestamp=`date +%Y%m%d%H`
main
