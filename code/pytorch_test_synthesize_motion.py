import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Condition_num=5
Groundtruth_num=5
In_frame_size = Joints_num*3


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
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
        
    
    #in cuda tensor initial_seq: b*(initial_seq_len*frame_size)
    #out cuda tensor out_seq  b* ( (intial_seq_len + generate_frame_number) *frame_size)
    def forward(self, initial_seq, generate_frames_number):
        batch = initial_seq.size()[0]

        # Initialize vec_h and vec_m
        (vec_h, vec_c) = self.init_hidden(batch)

        # Original sequence tensor
        original_seq = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch,1))).cuda())

        # Output sequence tensor
        out_seq = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch,1))).cuda())

        out_frame = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch,self.out_frame_size))).cuda())

        # Loop over the initial sequence
        for i in range(initial_seq.size()[1]):
            in_frame = initial_seq[:,i]

            # Store the initial sequence frames
            original_seq = torch.cat((original_seq, in_frame), 1)

            (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        # Loop over the frames to be generated
        for i in range(generate_frames_number):
            in_frame = out_frame

            (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        # Discard the first frame (placeholder) from the sequences
        original_seq = original_seq[:, 1: original_seq.size()[1]]
        out_seq = out_seq[:, 1: out_seq.size()[1]]

        return out_seq, original_seq

    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss


#numpy array inital_seq_np: batch*seq_len*frame_size
# return numpy b*generate_frames_number*frame_data
# def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):

#     #set hip_x and hip_z as the difference from the future frame to current frame
#     dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0: initial_seq_np.shape[1]-1]
#     initial_seq_dif_hip_x_z_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
#     initial_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
#     initial_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    
#     initial_seq  = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np.tolist()).cuda() )
 

#     # seq_len=real_seq.size()[1]-1
#     seq_len = initial_seq.size()[1]-1
#     # in_real_seq=real_seq[:, 0:seq_len]
#     in_real_seq=initial_seq[:, 0:seq_len]

#     predict_groundtruth_seq = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np[:,1:seq_len+1].tolist())).cuda().view(initial_seq_np.shape[0],-1)


#     predict_seq = model.forward(in_real_seq, generate_frames_number)
    
#     batch=initial_seq_np.shape[0]
#     # print(f'in generate_seq, batch is {batch}')
   
#     for b in range(batch):
#         # save the groundtruth_seq
#         groundtruth_seq = np.array(initial_seq_np[b].tolist()).reshape(-1,In_frame_size)
#         last_x=0.0
#         last_z=0.0
#         for frame in range(groundtruth_seq.shape[0]):
#             groundtruth_seq[frame,Hip_index*3]=groundtruth_seq[frame,Hip_index*3]+last_x
#             last_x=groundtruth_seq[frame,Hip_index*3]
            
#             groundtruth_seq[frame,Hip_index*3+2]=groundtruth_seq[frame,Hip_index*3+2]+last_z
#             last_z=groundtruth_seq[frame,Hip_index*3+2]

#         read_bvh.write_traindata_to_bvh(save_dance_folder+"groundtruth"+"%02d"%b+".bvh", groundtruth_seq)

#         out_seq=np.array(predict_seq[b].data.tolist()).reshape(-1,In_frame_size)
#         last_x=0.0
#         last_z=0.0
#         for frame in range(out_seq.shape[0]):
#             out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
#             last_x=out_seq[frame,Hip_index*3]
            
#             out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
#             last_z=out_seq[frame,Hip_index*3+2]
            
#         read_bvh.write_traindata_to_bvh(save_dance_folder+"out"+"%02d"%b+".bvh", out_seq)
#     return np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size)


def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):

    #set hip_x and hip_z as the difference from the future frame to current frame
    dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0: initial_seq_np.shape[1]-1]
    initial_seq_dif_hip_x_z_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    # print(f'in generate_seq -- initial_seq_np is {initial_seq_np.shape}') # (5, 14, 171)
    
    
    initial_seq  = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np.tolist()).cuda() )
    # print(f'in generate_seq -- initial_seq.shape is {initial_seq.shape}') # torch.Size([5, 14, 171])
    original_seq, predict_seq = model.forward(initial_seq, generate_frames_number)
    # print(f'in generate_seq -- predict_seq.shape is {predict_seq.shape}') #  torch.Size([5, 70794])

    batch=initial_seq_np.shape[0]  #(5, 15, 171) 5 is batch
    # print(f'in generate_seq, batch is {batch}')
    # print(f'out_seq. shape 0 is {out_seq.shape}')   

    # target_seq = np.array(initial_seq_np[:,1:initial_seq_np.shape[1]].tolist()).reshape(-1,In_frame_size)
    target_seq = initial_seq_np[:, 1:1+generate_frames_number] #(5, 14, 171) 14 generate_frames_number
    # print(f'generate_frames_number is {generate_frames_number}')
    # print(f'In generate seq -- target_seq.shape is {target_seq.shape}') 
    # target_seq = initial_seq_np[:, 1:initial_seq_np.shape[1] + generate_frames_number]

    for b in range(batch):    
        # Modify out_seq to be the same shape as target_seq
        out_seq = np.array(predict_seq[b].data.tolist()).reshape(-1, In_frame_size) #(:generate_frames_number) , frame size=171, 
        # print(f'target_seq.shape is {target_seq.shape}')
        # print(f'frame size is {In_frame_size}')
        # print(f'out_seq.shape is {out_seq.shape}') #(29, 171)

        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]): #414
            

            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]
            
            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]

        # print(f'out_seq.shape after FOR LOOP  is {out_seq.shape}') #  (414, 171)
        # Save the predicted sequence to a .bvh file
        read_bvh.write_traindata_to_bvh(save_dance_folder + "out" + "%02d" % b + ".bvh", out_seq)

        # Get the corresponding original sequence
        original_seq_np = np.array(original_seq[b].data.tolist()).reshape(-1, In_frame_size)

        # Save the original sequence to a .bvh file
        read_bvh.write_traindata_to_bvh(save_dance_folder + "original" + "%02d" % b + ".bvh", original_seq_np)

        print(f'predict_seq.data.tolist()).reshape(batch, -1, In_frame_size) shape is {np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size).shape}') #(5, 414, 171)
    # print(f'in generate_seq -- predict_seq.shapeAFTER FOR LOOP is {predict_seq.shape}') #  torch.Size([5, 70794])

    return np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size), target_seq   #shape is (5, 414, 171)



#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        length=len(dance)/100
        length=10
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
def test(dance_batch_np, frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path, write_bvh_motion_folder):
    
    torch.cuda.set_device(0)

    model = acLSTM()    
    
    model.load_state_dict(torch.load(read_weight_path))
    
    model.cuda()
    
    
    #dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    
    dance_batch=[]
    
    for b in range(batch):
        #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
        dance_id = dance_len_lst[np.random.randint(0,random_range)]
        # print(f'Picked dance_id {dance_id}')
        dance=dances[dance_id].copy()
        
        dance_len = dance.shape[0]
        # print(f'Dance length {dance_len}')
            
        start_id=random.randint(10, int(dance_len-initial_seq_len*speed-10))#the first and last several frames are sometimes noisy. 
        sample_seq=[]
        for i in range(initial_seq_len):
            sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
        
        dance_batch=dance_batch+[sample_seq]
            
    dance_batch_np=np.array(dance_batch)
    # print(f'In test -- dance_batch_np.shape {dance_batch_np.shape}') #(5, 15, 171) 
 
    predicted_seq, target_seq = generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)







        

# read_weight_path="../train_weight_aclstm_indian/0086000.weight"
# write_bvh_motion_folder="../test_bvh_aclstm_indian/"
# dances_folder = "../train_data_xyz/indian/"

# martial
read_weight_path = "./weights/weights_pos/2000/0014000.weight"
write_bvh_motion_folder = "./synthetic_motion/pos/"
dances_folder = "./train_data/pos_train_xyz/"


# # euler
# read_weight_path = "./weights/weights_euler/0011000.weight"
# write_bvh_motion_folder = "./synthetic_motion/euler"
# dances_folder = "./train_data/euler_train_xyz/"

# #6d
# read_weight_path = "./weights/weights_6d/007000.weight"
# write_bvh_motion_folder = "./synthetic_motion/6d/"
# dances_folder = "./train_data/6d_train_xyz/"

dance_frame_rate=60
batch=5
initial_seq_len=30
generate_frames_number=300

if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)

dances= load_dances(dances_folder)
# len(dances
# print(f'len(dances) is {len(dances)}')

test(dances, dance_frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,  write_bvh_motion_folder)



