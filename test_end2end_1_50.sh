cfgs_file='cfgs/anet_test.yml'
split='testing'
id=1
epoch=19
#dataset_file=/data6/users/xuran7/myresearch/dense_videocap/third_party/densecap/data/anet/anet_annotations_trainval.json
dataset_file=/data6/users/xuran7/myresearch/data/anet/captions/test_ids.json
feature_root=/data6/users/xuran7/myresearch/data/anet/end2end

CUDA_VISIBLE_DEVICES=1 python3 scripts/test.py \
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
--max_prop_num=50 \
--min_prop_num=10 \
--min_prop_before_nms=20 \
--densecap_res=./results/dense_cap_testing_1_50.json \
--prop_res=./results/prop_testing_1_50.json \
#| tee log/eval-$id-epoch$epoch
