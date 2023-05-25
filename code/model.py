import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
# import tensorflow as tf
import time

import torch.nn.functional as F

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Condition_num=5
Groundtruth_num=5
In_frame_size = Joints_num*3





class acLSTM(nn.Module):
    def __init__(self, in_frame_size=132, hidden_size=1024, out_frame_size=132):
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    
    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        return  ([h0,h1,h2], [c0,c1,c2])
    
    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        
        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2) #out b*150
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        
        batch=real_seq.size()[0]
        seq_len=real_seq.size()[1]
        
        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        
        
        for i in range(seq_len):
            
            if(condition_lst[i]==1):##input groundtruth frame
                in_frame=real_seq[:,i]
            else:
                in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
       
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_mse_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss
    
 
    # used for Euler angles  and 6D representation
    # def angle_distance_loss(self, out_seq, groundtruth_seq):
    #     # assert out_seq.shape == groundtruth_seq.shape, "The two sequences must have the same shape."
    #     # assert len(out_seq.shape) == 3 and out_seq.shape[-1] == 3, "The two sequences must be 3D tensors of unit vectors."
        
    #     # Calculate cosine similarity between the two vectors
    #     cosine_sim = F.cosine_similarity(out_seq, groundtruth_seq, dim=1)
    #     ad = torch.sum(1 - cosine_sim).mean()
    #     # Return Angle Distance as the loss
    #     return ad
    
    def angle_distance_loss(self, out_seq, groundtruth_seq):

        loss = torch.sum(1 - torch.cos(out_seq - groundtruth_seq)).mean()
        return loss
    

    def cal_loss(self, out_seq, groundtruth_seq):
        # print(f'batch size: {out_seq.shape[0]}')
        # print(f'sequence length: {out_seq.shape[1]}')
        # Seq_len = out_seq.shape[1] // 261

        output = out_seq.reshape(32, Seq_len, 261)
        hip_output = output[:, :, 0:3].reshape(32, Seq_len*3)
        euler_output = output[:, :, 3:].reshape(32*Seq_len*258)


        gt = groundtruth_seq.reshape(32, Seq_len, 261)
        hip_gt = gt[:, :, 0:3].reshape(32, Seq_len*3)
        euler_gt = gt[:, :, 3:].reshape(32*Seq_len*258)


        hip_loss = nn.MSELoss()
        angle_loss = self.angle_distance_loss(euler_output, euler_gt)

        mse_loss = hip_loss(hip_output, hip_gt)

        # total loss with weight
        # loss = 0.5 * mse_loss + 0.5 * angle_loss
        loss = mse_loss + angle_loss

        print("hip loss: ", mse_loss)
        print("angle loss: ", angle_loss)
        # print("total loss: ", loss)

        return loss

    
    def calculate_quaternion_loss(self, out_seq, groundtruth_seq):
        assert out_seq.shape == groundtruth_seq.shape, "The two sequences must have the same shape."
        assert len(out_seq.shape) == 3 and out_seq.shape[-1] == 4, "The two sequences must be 3D tensors of quaternions."

        q_out = out_seq.view(-1, 4)
        q_gt = groundtruth_seq.view(-1, 4)

        # Compute the cosine similarity along the last dimension
        cosine_sim = F.cosine_similarity(q_out, q_gt, dim=-1)

        # Compute the mean absolute error of the cosine similarity
        qloss = torch.mean(torch.abs(1 - cosine_sim))

        return qloss