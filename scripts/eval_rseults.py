# general packages
import os
import argparse
import numpy as np
from collections import defaultdict
import json
import subprocess
import csv
import yaml
import sys
import pdb

sys.path.insert(0, '/data6/users/xuran7/myresearch/dense_videocap/third_party/myfork/densecap')
sys.path.insert(0, '/data6/users/xuran7/myresearch/dense_videocap/evaluation/densevid_eval/coco-caption')

from data.utils import update_values
from tools.eval_proposal_anet import ANETproposal


parser = argparse.ArgumentParser()

parser.add_argument('--densecap_eval_file', 
default='/data6/users/xuran7/myresearch/dense_videocap/evaluation/densevid_eval/evaluate.py')
parser.add_argument('--densecap_res', default='./results/dense_cap_1_50.json', type=str)
parser.add_argument('--prop_res', default='./results/prop_1_50.json', type=str)
parser.add_argument('--cfgs_file', default='cfgs/anet.yml', type=str, help='dataset specific settings. anet | yc2')
parser.add_argument('--dataset_file', default='/data6/users/xuran7/myresearch/dense_videocap/third_party/myfork/densecap/data/anet/anet_annotations_trainval.json', type=str)
parser.add_argument('--val_data_folder', default='validation', help='validation data folder')


parser.set_defaults(cuda=False, learn_mask=False, gated_mask=False)

args = parser.parse_args()
#pdb.set_trace()
with open(args.cfgs_file, 'r') as handle:
    options_yaml = yaml.load(handle)
update_values(options_yaml, vars(args))
print(args)


def eval_results(args):

    subprocess.Popen(["python2", args.densecap_eval_file, "-s", \
                      args.densecap_res, \
                      "-v", "-r"] + \
                      args.densecap_references \
                      )


    anet_proposal = ANETproposal(args.dataset_file,
                                 args.prop_res,
                                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                 max_avg_nr_proposals=100,
                                 subset=args.val_data_folder, verbose=True, check_status=True)

    anet_proposal.evaluate()

    return anet_proposal.area

if __name__ == "__main__":
    recall_area = eval_results(args)
    print('proposal recall area: {:.6f}'.format(recall_area))
