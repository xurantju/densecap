batch_size=24
mask_weight=1.0
sent_weight=0.25
cfgs_file='cfgs/anet.yml'
world_size=1
dist_url='none'
id=1
dataset_file=/data6/users/xuran7/myresearch/dense_videocap/third_party/myfork/densecap/data/anet/anet_annotations_trainval.json
feature_root=/data6/users/xuran7/myresearch/data/anet/end2end
dur_file=/data6/users/xuran7/myresearch/dense_videocap/third_party/myfork/densecap/data/anet/anet_duration_frame.csv

CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py \
--dist_url $dist_url \
--cfgs_file $cfgs_file \
--checkpoint_path ./checkpoint/$id \
--batch_size $batch_size \
--world_size $world_size \
--cuda \
--dataset=anet \
--feature_root=${feature_root} \
--sent_weight $sent_weight \
--mask_weight $mask_weight \
--dataset_file=${dataset_file} \
--dur_file=${dur_file} \
--gated_mask \
--learning_rate=0.01 \
#| tee log/$id-0 &
