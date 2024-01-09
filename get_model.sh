directory='./ExpressiveEncoding/third_party/models_back'

mkdir -p $directory

files_list=(79999_iter.pth e4e_ffhq_encode.pt model_ir_se50.pth modellarge10k.pt stylegan2_ffhq.pkl)


for file in ${files_list[@]};
do
    wget https://amemori-model.s3.cn-northwest-1.amazonaws.com.cn/models/$file --directory-prefix=${directory}
    echo $file" has downloaded into "${directory}
done
