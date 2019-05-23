cfgs_file='cfgs/anet.yml'
split='validation'
id=1
epoch=19
dataset_file=/data6/users/xuran7/myresearch/dense_videocap/third_party/myfork/densecap/data/anet/anet_annotations_trainval.json
feature_root=/data6/users/xuran7/myresearch/data/anet/end2end

CUDA_VISIBLE_DEVICES=2 python3 scripts/test.py \
--cfgs_file $cfgs_file \
--densecap_eval_file /data6/users/xuran7/myresearch/dense_videocap/evaluation/densevid_eval/evaluate.py \
--batch_size 1 \
--start_from ./checkpoint/$id/model_epoch_$epoch.t7 \
--id $id-$epoch \
--val_data_folder $split \
--learn_mask --gated_mask --cuda \
--dataset_file=${dataset_file} \
--feature_root=${feature_root} \
--num_workers=8 \
#| tee log/eval-$id-epoch$epoch
