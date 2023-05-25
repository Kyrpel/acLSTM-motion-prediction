import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
# import tensorflow as tf
import sys
sys.path.append("code")
import time
import read_bvh


import torch.nn.functional as F

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
# Joints_num = 43
Condition_num=5
Groundtruth_num=5
In_frame_size = Joints_num*3



# In_frame_size = 132 for euler, 171 for positional, 261 for 6d
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
    def calculate_mse_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss
    
    def cal_loss(self, out_seq, groundtruth_seq):
        output = out_seq.reshape(32, Seq_len, 132)
        hip_output = output[:, :, 0:3].reshape(32, Seq_len*3)
        euler_output = output[:, :, 3:].reshape(32*Seq_len*129)


        gt = groundtruth_seq.reshape(32, Seq_len, 132)
        hip_gt = gt[:, :, 0:3].reshape(32, Seq_len*3)
        euler_gt = gt[:, :, 3:].reshape(32*Seq_len*129)


        mse = nn.MSELoss()

        hip_loss = mse(hip_output, hip_gt)
        euler_loss = self.angle_distance_loss(euler_output, euler_gt)  

        # total loss with weight
        hip_loss_weight = 0.7  # assign weight to hip_loss
        euler_loss_weight = 0.3  # assign weight to euler_loss

        # total loss with weight
        loss = hip_loss_weight * hip_loss + euler_loss_weight * euler_loss

        print("hip loss: ", hip_loss)
        print("euler loss: ", euler_loss)
        # print("total loss: ", loss)

        return loss


    
    def angle_distance_loss(self, out_seq, groundtruth_seq):

        loss = torch.sum(1 - torch.cos(out_seq - groundtruth_seq)) / (32*100)
        return loss
    
    





def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True):
    # print(f'SHA of real_seq_np: {real_seq_np.shape}')
    # read_bvh.write_euler_traindata_to_bvh(save_dance_folder+"Real_seq_np_reshape.bvh", real_seq_np)
    #set hip_x and hip_z as the difference from the future frame to current frame
    dif = real_seq_np[:, 1:real_seq_np.shape[1]] - real_seq_np[:, 0: real_seq_np.shape[1]-1]
    real_seq_dif_hip_x_z_np = real_seq_np[:, 0:real_seq_np.shape[1]-1].copy()
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    real_seq  = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np.tolist()).cuda() )
 
    seq_len=real_seq.size()[1]-1
    in_real_seq=real_seq[:, 0:seq_len]
    
    
    predict_groundtruth_seq= torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np[:,1:seq_len+1].tolist())).cuda().view(real_seq_np.shape[0],-1)
   
   
    
    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)
    
    optimizer.zero_grad()
    
    loss=model.cal_loss(predict_seq, predict_groundtruth_seq)
    
    loss.backward()
    
    optimizer.step()
    
    if(print_loss==True):
        print ("###########"+"weight loss Euler iter %07d"%iteration +"######################")
        print ("loss: "+str(loss.data.tolist()))

    
    if(save_bvh_motion==True):
        ##save the first motion sequence int the batch.
        gt_seq=np.array(predict_groundtruth_seq[0].data.tolist()).reshape(-1,132)
        # print("gt_seq.shape: ", gt_seq.shape) #(100, 132)
        # read_bvh.write_euler_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"TEST_GT.bvh", gt_seq)
        last_x=0.0
        last_z=0.0
        for frame in range(gt_seq.shape[0]):
            gt_seq[frame,Hip_index*3]=gt_seq[frame,Hip_index*3]+last_x
            last_x=gt_seq[frame,Hip_index*3]
            
            gt_seq[frame,Hip_index*3+2]=gt_seq[frame,Hip_index*3+2]+last_z
            last_z=gt_seq[frame,Hip_index*3+2]
        
        out_seq=np.array(predict_seq[0].data.tolist()).reshape(-1,132)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]
            
            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]
            
        
        read_bvh.write_euler_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_gt.bvh", gt_seq)
        read_bvh.write_euler_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_out.bvh", out_seq)

#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        #length=len(dance)/100
        length = 10
        if(length<1):
            length=1              
        len_lst=len_lst+[length]
    
    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length):
            index_lst=index_lst+[index]
        index=index+1
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    for dance_file in dance_files:
        print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file)
        print ("frame number: "+ str(dance.shape[0]))
        dances=dances+[dance]
    return dances
    
# dances: [dance1, dance2, dance3,....]
def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder, write_bvh_motion_folder, total_iter=500000):
    
    seq_len=seq_len+2
    torch.cuda.set_device(0)
    # check if cuda is available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("using cuda")
    else:
        print("using cpu")

    model = acLSTM()    
    
    if(read_weight_path!=""):
        model.load_state_dict(torch.load(read_weight_path))
    
    model.cuda()
    # model=torch.nn.DataParallel(model, device_ids=[0,1])

    current_lr=0.0001
    # optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    # for part B -  expiramneting
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9)

    
    model.train()
    
    #dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    # print ("random range: "+str(random_range))
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    for iteration in range(total_iter):   
        #get a batch of dances
        dance_batch=[]
        for b in range(batch):
            #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
            dance_id = dance_len_lst[np.random.randint(0,random_range)]
            dance=dances[dance_id].copy()
            # print(f'dance shape: {dance.shape}') #(6117, 132), (3262, 132) ..
            
            dance_len = dance.shape[0]
            
            start_id=random.randint(10, dance_len-seq_len*speed-10)#the first and last several frames are sometimes noisy. 
            sample_seq=[]
            for i in range(seq_len):
                sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
            
            # #augment the direction and position of the dance
            # T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
            # R=[0,1,0,(random.random()-0.5)*np.pi*2]
            # sample_seq_augmented=read_bvh.augment_train_data(sample_seq, T, R)
            # print(f'sample_seq_augmented shape: {sample_seq_augmented.shape}')

            dance_batch=dance_batch+[sample_seq]
        dance_batch_np=np.array(dance_batch)
        # print(f'dance_batch_np shape: {dance_batch_np.shape}') # (32, 102, 132)
        # dance_batch_np=dance_batch_np.reshape(-1,132)
        # print(f'dance_batch_np shape: {dance_batch_np.shape}') #(3264, 132)
        # print(f'dance_batch_np : {dance_batch_np}')
        # read_bvh.write_euler_traindata_to_bvh('DDance_batch.bvh', dance_batch_np)
        
        print_loss=False
        save_bvh_motion=False
        if(iteration % 1==0):
            print_loss=True
        if(iteration % 1000==0):
            save_bvh_motion=True
            
        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss, save_bvh_motion)
        # break
        end=time.time()

        #print end-start
        if(iteration%1000 == 0):
            path = write_weight_folder + "%07d"%iteration +".weight"
            torch.save(model.state_dict(), path)
        

        # commnet out each one of the following lines to train the network if you  dont want to use them
if os.path.exists("/kaggle/input/auto-rnn"):#if we are running on kaggle
    print("on train py running on KAGGLE")
    # for positional
    # read_weight_path=""
    # write_weight_folder = "kaggle/working/weights/weights_pos/"
    # write_bvh_motion_folder = "kaggle/working/pos_train_tmp_bvh_aclstm_martial/"
    # dances_folder = "/kaggle/input/auto-rnn/train_data/pos_train_xyz/"



    # for euler
    # read_weight_path="./weights/weights_euler/0007000.weight"
    # # location to save the weights of the network during training
    # write_weight_folder="/kaggle/working/weights/weights_euler/7000/"
    # # location to save the temporate output of the network and the groundtruth motion sequences in the form of bvh
    # write_bvh_motion_folder="/kaggle/working/euler_train_tmp_bvh_aclstm_martial/6000/"
    # # location of the training data
    # dances_folder = "/kaggle/input/auto-rnn/train_data/euler_train_xyz/"



else: #if we are running on local machine
    print("on train py running on LOCAL machine - GD")


    # for euler
    read_weight_path=""
    # location to save the weights of the network during training
    write_weight_folder="./weights/weights_euler/weight_loss_weights/"
    #  location to save the temporate output of the network and the groundtruth motion sequences in the form of bvh
    write_bvh_motion_folder="./tmp/weight_loss_euler_train_tmp/"
    # location of the training data
    dances_folder = "./train_data/euler_train_xyz/"






dance_frame_rate=60
batch=32

if not os.path.exists(write_weight_folder):
    os.makedirs(write_weight_folder)
if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)
    

dances= load_dances(dances_folder)


start_time = time.time()
train(dances, dance_frame_rate, batch, 100, read_weight_path, write_weight_folder, write_bvh_motion_folder, 7000)


print ("total time: "+str(time.time()-start_time))


