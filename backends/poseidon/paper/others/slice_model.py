import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Subset
# 假设这些是你项目中存在的库
from poseidon_main.scOT.inference import get_trainer, rollout
from poseidon_main.scOT.problems.base import get_dataset
 
from MySVD.helper.input_collector import *
from MySVD.helper.rank_selector import *
from MySVD.helper.fisher_collector import *
from MySVD.helper.model_slicer import *

from MySVD.compression_method.compress_dobi_rank import svd_compress_sliced_pairwise_dobirank
from MySVD.compression_method.compress_fixed_rank import svd_compress_sliced_pairwise_fixedrank, svd_compress_sliced_pairwise_incremental_rank
from MySVD.compression_method.keep_full_rank import svd_reconstruct_full_rank_physics_aware
from MySVD.compression_method.compress_pure_svd import svd_compress_standard_puresvd
device = 'cuda'

energy_keep = 0.999
time_step = "0to10"
model = "Poseidon-L"
model_type = "finetune"
downsteam_task = "fluids.incompressible.BrownianBridge"
use_phyloss = "no_phyloss"
order_control = "0"
gradient_loss = "mse" # mse or l1
rank_mode = "fixed_rank" # energy_rank / dobi_rank / fixed_rank / full_rank / incremental_rank
use_fisher = "y"

cali_data = get_dataset(
    dataset=downsteam_task,
    which="train",
    num_trajectories=128,
    data_path=f"poseidon_main/camlab-ethz/down_streams",
    fix_input_to_time_step=None,
    time_step_size=1, 
    max_num_time_steps=20, 
)

trainer = get_trainer(
    model_path=f"checkpoints/{model}/{use_phyloss}/{downsteam_task}/origin/none",
    batch_size=256,
    dataset=cali_data,
)
model = trainer.model


print(cali_data)
print(type(cali_data))
print(len(cali_data))
part_data = Subset(cali_data, [1,10])
rollout(trainer, part_data, model_path="Poseidon-L")
# for data in cali_data:
#     print(data.keys())