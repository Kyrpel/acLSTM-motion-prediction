import sys
sys.path.append("code")

# import nn
import torch.nn as nn
import numpy as np
import os
import torch
from generate_training_data import generate_traindata_from_bvh


Seq_len=100

def calculate_mse_loss( out_seq, groundtruth_seq):


        loss_function = nn.MSELoss()
        mse = loss_function(out_seq, groundtruth_seq)
        return mse.item()



def proccess(representation):
    if representation == '6d':
        print("6d to positional - calculate the MSE")
        # This is the directory where the original sequence files are stored
        src_bvh_folder = './tmp/6d_train_tmp_bvh_aclstm_martial/'
        # This is the directory where we will store the generated tranning data 
        tar_traindata_folder = './tmp/eval_tmp/6dBVH_to_pos/'
        if not os.path.exists(tar_traindata_folder):
            os.makedirs(tar_traindata_folder)
        generate_traindata_from_bvh(src_bvh_folder, tar_traindata_folder, representation='positional')
        
    elif representation == 'euler_angles':
        print("Converting Eulers angles to positional - calculate the MSE")
        src_bvh_folder = './tmp/euler_train_tmp_bvh_aclstm_martial/'
        tar_traindata_folder = './tmp/eval_tmp/eulerBVH_to_pos/'
        if not os.path.exists(tar_traindata_folder):
            os.makedirs(tar_traindata_folder)
        generate_traindata_from_bvh(src_bvh_folder, tar_traindata_folder, representation='positional')

    elif representation == 'positional':
        print("positional data - calculate the MSE")

        src_bvh_folder = './tmp/train_tmp_bvh_aclstm_martial/'
        tar_traindata_folder = './tmp/eval_tmp/posBVH_to_pos/'
        if not os.path.exists(tar_traindata_folder):
            os.makedirs(tar_traindata_folder)
        generate_traindata_from_bvh(src_bvh_folder, tar_traindata_folder, representation='positional')
        #move to proccessign the positional data directly and compute the loss - the trainning data is already in the directory

    else:
        #  quaternion
        print("Converting quaternions to positional - calculate the MSE")
        src_bvh_folder = './tmp/quar_train_tmp_bvh_aclstm_martial/'
        # is the dir don not exist, create it
        tar_traindata_folder = './tmp/eval_tmp/quarBVH_to_pos/'
        if not os.path.exists(tar_traindata_folder):
            os.makedirs(tar_traindata_folder)
        generate_traindata_from_bvh(src_bvh_folder, tar_traindata_folder, representation='positional')
    return src_bvh_folder, tar_traindata_folder




















    #  ********* change the representation parameter to the one you want to evaluate, 6d, euler_angles or quartenions or positional *********
# _, tar_traindata_folder = proccess('euler_angles')
_, tar_traindata_folder = proccess('quartenions')
# Convert bvh files to positional fisrt and then calsulate the MSE loss


print("-----------------------------------")
print("Calculate the MSE loss for the generated positional data")
# Iterate over the files in the newly generated training data folder
for file in os.listdir(tar_traindata_folder):
    # Check if it's a ground truth file
    if file.endswith('_gt.bvh.npy'):
        # Get the corresponding output file
        out_file = file.replace('_gt.bvh.npy', '_out.bvh.npy')
        print(f"out_file: {out_file}")
        print(f"file: {file}")
        
        # Check if the corresponding output file exists
        if out_file in os.listdir(tar_traindata_folder):
            # Load the ground truth and output sequences
            gt_seq = torch.from_numpy(np.load(os.path.join(tar_traindata_folder, file)))
            out_seq = torch.from_numpy(np.load(os.path.join(tar_traindata_folder, out_file)))
            
            # Calculate the loss
            loss_value = calculate_mse_loss(out_seq, gt_seq)
            
            # Remove '_gt.bvh.npy' from the file name
            file_name = file.replace('_gt.bvh.npy', '')
            
            # Print the loss value with the specified iteration weight model
            print(f"Loss for {file_name} iteration weight model: {loss_value:.6f}")



