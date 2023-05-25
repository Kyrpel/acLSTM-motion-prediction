import read_bvh
import numpy as np
from os import listdir
import os

def generate_traindata_from_bvh(src_bvh_folder, tar_traindata_folder, representation='euler_angles'):
    '''
    representation = 'positional' or 'euler_angles' or  '6d' or 'quaternion'
    '''

    print("Generating training data for " + src_bvh_folder)
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names = os.listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len = len(bvh_dance_name)
        if name_len > 4 and bvh_dance_name[name_len - 4: name_len] == ".bvh":
            print("Processing " + bvh_dance_name)
            if representation == 'positional':
                dance = read_bvh.get_train_data(src_bvh_folder + bvh_dance_name)
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", dance)
            if representation == 'euler_angles':
                dance = read_bvh.get_training_data_euler_angles(src_bvh_folder + bvh_dance_name)
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", dance)
            if representation == '6d':
                dance = read_bvh.get_training_data_6d(src_bvh_folder + bvh_dance_name)
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", dance)
            if representation == 'quaternions':
                dance = read_bvh.get_training_data_quaternions(src_bvh_folder + bvh_dance_name)
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", dance)





def generate_bvh_from_traindata(src_train_folder, tar_bvh_folder, representation='6d'):
    
    print ("Generating bvh data for "+ src_train_folder)
    if (os.path.exists(tar_bvh_folder)==False):
        os.makedirs(tar_bvh_folder)
    dances_names=listdir(src_train_folder)
    for dance_name in dances_names:
        name_len=len(dance_name)
        if(name_len>4):
            if(dance_name[name_len-4: name_len]==".npy"):
                print ("Processing"+dance_name)
                dance=np.load(src_train_folder+dance_name)
                dance2=[]
                print(f'dance.shape: {dance.shape}') # (6117, 132)
                # print(f'dance.shape: {dance.shape}')
                for i in range(dance.shape[0]//8):
                    dance2=dance2+[dance[i*8]]
                print (len(dance2))
                if representation == 'positional':
                    print('writing positional')
                    # print(f'dance2.shape: {np.array(dance2).shape}') #(764, 171)
                    # print(f'dance2: {dance2}')
                    read_bvh.write_traindata_to_bvh(tar_bvh_folder+dance_name+".bvh",np.array(dance2))
                if representation == 'euler_angles':
                    print('writing euler angles')
                    read_bvh.write_euler_traindata_to_bvh(tar_bvh_folder+dance_name+".bvh", np.array(dance2))
                if representation == '6d':
                    print('writing 6d')
              
                    read_bvh.write_6d_traindata_to_bvh(tar_bvh_folder+dance_name+".bvh", np.array(dance2))
                if representation == 'quaternions':
                    print('writing quaternions')
                    read_bvh.write_quaternion_traindata_to_bvh(tar_bvh_folder+dance_name+".bvh",np.array(dance2))
                


# generate_traindata_from_bvh("./train_data_bvh/indian/","./train_data_xyz/indian/")
# generate_traindata_from_bvh("./train_data_bvh/salsa/","./train_data_xyz/salsa/")

# generate_traindata_from_bvh("./train_data_bvh/martial/","./train_data_xyz/martial_eul/", 'euler_angles')

# generate_traindata_from_bvh("./train_data_bvh/martial/","./train_data_xyz/martial_quaternions/", 'quaternions')



#************************ mote:  you have to comment out the bellow if you are running the loss evaluation python file


# testing
# positional
# generate_traindata_from_bvh("./train_data_bvh/martial/","./train_data_xyz/test_pos/", "positional")
# euler angles
# generate_traindata_from_bvh("./train_data_bvh/martial/","./train_data/euler_train_xyz/", 'euler_angles')
# 6d
# generate_traindata_from_bvh("./train_data_bvh/martial/","./6d_train_xyz/", '6d')
# quaternions
# generate_traindata_from_bvh("./train_data_bvh/martial/","./train_data/quartentions_train_xyz/", 'quaternions')






# generate_bvh_from_traindata("./train_data/pos_train_xyz/","./train_data_bvh/test_pos/", 'positional')
# 
# generate_bvh_from_traindata("./train_data/euler_train_xyz/","./train_data_bvh/test_eur/", 'euler_angles')
# for 6d
# generate_bvh_from_traindata("./train_data/6d_train_xyz/","./train_data_bvh/test_6d/", '6d')
# for quaternions
# generate_bvh_from_traindata("./train_data/quartentions_train_xyz/","./train_data_bvh/test_quaternions/", 'quaternions')