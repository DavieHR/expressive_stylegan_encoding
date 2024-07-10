set -e
python -m ExpressiveEncoding.validate \
       --save_path `pwd`/results/ \
       --latest_decoder_path `pwd`/results/pivot_027/snapshots/100.pth \
       --stage_three_path `pwd`/results/exp010/0/pose \
       --face_folder_path `pwd`/results/exp010/0/data/smooth \
       --attribute_path `pwd`/results/attribute_finetuning_002/expressive
       #--attribute_path `pwd`/results/exp010/0/expressive
