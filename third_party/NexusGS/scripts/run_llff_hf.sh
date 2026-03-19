export CUDA_VISIBLE_DEVICES=$1

name=fern
n=3
dataset=./hf_models/NexusGS-llff/$name
# dataset=Yukinoo/NexusGS-llff
revision=$name
workspace=output/llff/$name/${n}_views
iterations=30000
dataset_type=llff
images=images_8
split_num=4
valid_dis_threshold=1.0
drop_rate=1.0
near_n=2

python train.py --source_path $dataset --model_path $workspace --eval --n_views $n \
    --save_iterations  30000 \
    --iterations $iterations \
    --densify_until_iter $iterations \
    --position_lr_max_steps $iterations \
    --dataset_type $dataset_type \
    --images $images \
    --split_num $split_num \
    --valid_dis_threshold $valid_dis_threshold \
    --drop_rate $drop_rate \
    --near_n $near_n \
    --huggingface \
    --revision $revision

python render.py --source_path $dataset  --model_path  $workspace --iteration 30000 --render_depth --huggingface


python metrics.py --source_path $dataset --model_path $workspace 