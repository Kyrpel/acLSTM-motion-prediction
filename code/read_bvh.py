import sys
sys.path.append("code")
sys.path.append("train_data_bvh")
import numpy as np
import cv2 
from cv2 import VideoCapture
import matplotlib.pyplot as plt
from collections import Counter
# import transforms3d
# import transforms3d.euler as euler
import transforms3d.quaternions as quat

from pylab import *
from PIL import Image
import os
import getopt

import json # For formatted printing

import read_bvh_hierarchy

import rotation2xyz as helper
from rotation2xyz import *
from scipy.spatial.transform import Rotation as R

# extra functions for converting 
import extra

import torch






def get_pos_joints_index(raw_frame_data, non_end_bones, skeleton):
    pos_dic=  helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    keys=OrderedDict()
    i=0
    for joint in pos_dic.keys():
        keys[joint]=i
        i=i+1
    return keys


def parse_frames(bvh_filename):
   bvh_file = open(bvh_filename, "r")
   lines = bvh_file.readlines()
   bvh_file.close()
#    find the line that contain the text "MOTION"
   l = [lines.index(i) for i in lines if 'MOTION' in i]
   data_start=l[0]
   #data_start = lines.index('MOTION\n')
   first_frame  = data_start + 3 # this is because the first frame is always 3 lines after the line that says "MOTION"
   
   num_params = len(lines[first_frame].split(' ')) 
   num_frames = len(lines) - first_frame
                                     
   data= np.zeros((num_frames,num_params))
#     raw values
   for i in range(num_frames):
       line = lines[first_frame + i].split(' ')
       line = line[0:len(line)]

       
       line_f = [float(e) for e in line]
       
       data[i,:] = line_f
           
   return data

# thesea are global strucrures
if os.path.exists("./train_data_bvh/standard.bvh"):
    standard_bvh_file="./train_data_bvh/standard.bvh"
else: #on kaggle
    standard_bvh_file = "/kaggle/input/auto-rnn/train_data_bvh/standard.bvh"

weight_translation=0.01
skeleton, non_end_bones=read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)    
sample_data=parse_frames(standard_bvh_file)
joint_index= get_pos_joints_index(sample_data[0],non_end_bones, skeleton)

   
def get_frame_format_string(bvh_filename):
    bvh_file = open(bvh_filename, "r")
    lines = bvh_file.readlines()
    bvh_file.close()
    l = [lines.index(i) for i in lines if 'MOTION' in i]
    data_end=l[0]
    #data_end = lines.index('MOTION\n')
    data_end = data_end+2
    return lines[0:data_end+1]

def get_min_foot_and_hip_center(bvh_data):
    print (bvh_data.shape)
    lowest_points = []
    hip_index = joint_index['hip']
    left_foot_index = joint_index['lFoot']
    left_nub_index = joint_index['lFoot_Nub']
    right_foot_index = joint_index['rFoot']
    right_nub_index = joint_index['rFoot_Nub']
                
                
    for i in range(bvh_data.shape[0]):
        frame = bvh_data[i,:]
        #print 'hi1'
        foot_heights = [frame[left_foot_index*3+1],frame[left_nub_index*3+1],frame[right_foot_index*3+1],frame[right_nub_index*3+1]]
        lowest_point = min(foot_heights) + frame[hip_index*3 + 1]
        lowest_points.append(lowest_point)
        
                                
        #print lowest_point
    lowest_points = sort(lowest_points)
    num_frames = bvh_data.shape[0]
    quarter_length = int(num_frames/4)
    end = 3*quarter_length
    overall_lowest = mean(lowest_points[quarter_length:end])
    
    return overall_lowest

def sanity():
    for i in range(4):
        print ('hi')
        
 
def get_motion_center(bvh_data):
    center=np.zeros(3)
    for frame in bvh_data:
        center=center+frame[0:3]
    center=center/bvh_data.shape[0]
    return center
 
    
from scipy.spatial.transform import Rotation

def  augment_train_frame_data(train_frame_data, T, axisR) :
    
    hip_index=joint_index['hip']
    hip_pos=train_frame_data[hip_index*3 : hip_index*3+3]
    
    for i in range(int(len(train_frame_data)/3) ):
        if(i!=hip_index):
            train_frame_data[i*3: i*3+3]=train_frame_data[i*3: i*3+3]+hip_pos
    
    
    mat_r_augment=euler.axangle2mat(axisR[0:3], axisR[3])
    n=int(len(train_frame_data)/3)
    for i in range(n):
        raw_data=train_frame_data[i*3:i*3+3]
        new_data = np.dot(mat_r_augment, raw_data)+T
        train_frame_data[i*3:i*3+3]=new_data
    
    hip_pos=train_frame_data[hip_index*3 : hip_index*3+3]
    
    for i in range(int(len(train_frame_data)/3)):
        if(i!=hip_index):
            train_frame_data[i*3: i*3+3]=train_frame_data[i*3: i*3+3]-hip_pos
    
    return train_frame_data

def augment_train_data(train_data, T, axisR):
    # print('augmenting data')
    result=list(map(lambda frame: augment_train_frame_data(frame, T, axisR), train_data))
    return np.array(result)



import numpy as np
import math

def extract_euler_angles(raw_frame_data, non_end_bones, skeleton):
    # Initialize an empty array to store the Euler angles for each joint
    euler_angles = np.zeros(len(non_end_bones) * 3)

    # Iterate through the joints in the skeleton
    for i, joint in enumerate(non_end_bones):
        # Extract the global transformation matrix for the current joint from the raw_frame_data
        global_transform = get_global_transform(joint, skeleton, raw_frame_data, non_end_bones)

        # Extract the rotation matrix from the global transformation matrix
        rotation_matrix = global_transform[:3, :3]
        # print(f' in extract_euler_angles ro tation matrix: {rotation_matrix})  


        # Convert the rotation matrix to Euler angles using the rotationMatrixToEulerAngles function
        euler_angles[i * 3:i * 3 + 3] = rotationMatrixToEulerAngles(rotation_matrix)

    # Modify the hip_pos initialization to be compatible with the data_vec_to_position_dic function
    hip_pos = np.zeros(3)
    hip_pos[:] = euler_angles[joint_index['hip'] * 3:joint_index['hip'] * 3 + 3]

    return euler_angles



#3rd extract_6d
def extract_6d(raw_frame_data, non_end_bones, skeleton):
    
    def get_rotation_matrix(joint, motion, skel):
        if joint == 'hip':
            return helper.eulerAnglesToRotationMatrix_hip(motion[3:6])
        else:
            joint_idx = non_end_bones.index(joint)
            theta_xyz = motion[6 + joint_idx * 3 : 6 + (joint_idx+1) * 3]
            return helper.eulerAnglesToRotationMatrix(theta_xyz[::-1])  # Reverse order to match the instructions

    six_d = np.zeros(len(non_end_bones) * 6)
    
    for i, joint in enumerate(non_end_bones):
        rotation_matrix = get_rotation_matrix(joint, raw_frame_data, skeleton)
        six_d[i*6:i*6+3] = rotation_matrix[0, :]  # First row (X-axis)
        six_d[i*6+3:i*6+6] = rotation_matrix[1, :]  # Second row (Y-axis)
    
    return six_d
    

def extract_quaternion(raw_frame_data, non_end_bones, skeleton):
    pos_dict = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    rotation_matrices, _ = helper.xyz_to_rotations_debug(skeleton, pos_dict)
    
    quaternion = np.zeros(len(non_end_bones) * 4)
    
    for i, joint in enumerate(non_end_bones):
        rotation_matrix = rotation_matrices[joint]
        quat = transforms3d.quaternions.mat2quat(rotation_matrix)
        quaternion[i*4:i*4+4] = quat
    
    return quaternion


# 2nd get_one_frame_training_format_data
def get_one_frame_training_format_data(raw_frame_data, non_end_bones, skeleton):
    
    # Default to positional representation
    pos_dic = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    new_data = np.zeros(len(pos_dic.keys())*3)
    i=0
    hip_pos = pos_dic['hip']
    for joint in pos_dic.keys():
        if(joint=='hip'):
            new_data[i*3:i*3+3] = pos_dic[joint].reshape(3)
        else:
            new_data[i*3:i*3+3] = pos_dic[joint].reshape(3) - hip_pos.reshape(3)
        i=i+1
    new_data = new_data * 0.01
    return new_data

import extra

def get_one_frame_training_format_data_euler_angles(raw_frame_data, non_end_bones, skeleton):
    # print("get_one_frame_training_format_data_euler_angles")
    rot_dic = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    new_data = np.zeros(len(rot_dic.keys()) * 3)
    # print(f'rot_dic', rot_dic)
    i = 0
    for joint in rot_dic.keys():
        # if joint == 'hip':
            # take the euler angles from the hip   CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
            # print(f'rot_dic[joint]: {rot_dic[joint]}')
            #   CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
            # print(f'rot_dic[joint]', rot_dic[joint])
            # new_data[i * 3:i * 3 + 3] = rot_dic[joint][3:6].reshape(3)
            # print(f'new_data for hip: {rot_dic[joint][3:6].reshape(3)}')
            # print(f'new_data for hip: {rot_dic[joint][3:].reshape(3)}')
            
        new_data[i * 3:i * 3 + 3] = rot_dic[joint].reshape(3)
        # print(f'new_data for {joint}: {rot_dic[joint].reshape(3)}')
        i = i + 1

    #convert new data to radians
    new_data = new_data * np.pi / 180
    # print(f'new_data: {new_data}')
    return new_data

import re

def get_one_frame_training_format_data_6d(raw_frame_data, non_end_bones, skeleton):
    # print("get_one_frame_training_format_data_6d--------")

    pos_dic = helper.get_skeleton_position(raw_frame_data, non_end_bones, skeleton)
    new_data = np.zeros(len(pos_dic.keys())*6)
    # print(f'len new data {len(new_data)}')
    i=0
    for joint in pos_dic.keys():
        if joint == 'hip':
            # print("hip")
            euler_angles = pos_dic[joint]
            # print(f'euler_angles {euler_angles}')
            rotation_matrix = eulerAnglesToRotationMatrix_hip(euler_angles)
            rotation_matrix_torch = torch.from_numpy(rotation_matrix)   
            six_d_rep = extra.matrix_to_rotation_6d(rotation_matrix_torch)  

        # find joint lRing2_Nub
        # elif joint == 'lRing2_Nub':
        #     print(f'pos_dic[lRing2_Nub] {pos_dic[joint]}')

        euler_angles = pos_dic[joint]
        rotation_matrix = eulerAnglesToRotationMatrix(euler_angles)
        rotation_matrix_torch = torch.from_numpy(rotation_matrix)
        six_d_rep = extra.matrix_to_rotation_6d(rotation_matrix_torch)
        # print(f'six_d_rep {six_d_rep}')
        new_data[i*6:i*6+6] = six_d_rep.numpy()
        # print(f' six_d_rep.numpy().reshape(6) {six_d_rep.numpy()}')

        i=i+1
    return new_data




# 2nd get training_format_data
def get_training_format_data(raw_data, non_end_bones, skeleton, representation='positional'):
    new_data=[]
    for frame in raw_data:
       
        new_frame=get_one_frame_training_format_data(frame, non_end_bones, skeleton)
    
        new_data=new_data+[new_frame]
    return np.array(new_data)



def get_weight_dict(skeleton):
    weight_dict=[]
    for joint in skeleton:
        parent_number=0.0
        j=joint
        while (skeleton[joint]['parent']!=None):
            parent_number=parent_number+1
            joint=skeleton[joint]['parent']
        weight= pow(math.e, -parent_number/5.0)
        weight_dict=weight_dict+[(j, weight)]
    return weight_dict

# def get_train_data(bvh_filename):
    
#     data=parse_frames(bvh_filename)
#     train_data=get_training_format_data(data, non_end_bones,skeleton)
#     center=get_motion_center(train_data) #get the avg position of the hip
#     center[1]=0.0 #don't center the height

#     new_train_data=augment_train_data(train_data, -center, [0,1,0, 0.0])
#     return new_train_data

# 2nd get_train_data
def get_train_data(bvh_filename):
    print(f'---------------------------get_positional train_data: raw_data')
  
    data = parse_frames(bvh_filename)
   
    train_data = get_training_format_data(data, non_end_bones, skeleton)
    # print(f'train_data: {train_data}')
    # print(f'len train_data: {len(train_data)}')

    center = get_motion_center(train_data)
    center[1] = 0.0
    new_train_data = augment_train_data(train_data, -center, [0, 1, 0, 0.0])

    return new_train_data #posinional procceed data






def return_euler_angle(data):
    new_data = []
    for frame in data:

        frame[3:] = frame[3:] * np.pi / 180  # Convert to radians
        new_data.append(frame)

    return np.array(new_data)


def get_training_data_euler_angles(bvh_filename):
    print(f'---------------------------get_training_format_data_euler_angles: raw_data----------')
     
    data = parse_frames(bvh_filename)
    
    # print(f'data shape: {data.shape}')
    # print(f'data: {data}')


    euler_angles_data = return_euler_angle(data)
    euler_angles_data[:, :3] *= 0.01  # normilize  the hip 

    # center = get_motion_center(euler_angles_data)
    # center[1] = 0.0
    # new_train_data = augment_train_data(euler_angles_data, -center, [0, 1, 0, 0.0])

    # print(f'euler_angles_data shape: {euler_angles_data.shape}')
    # print(f'HIP euler_angles_data: {euler_angles_data[:, :3]}')
    

    return euler_angles_data

def get_training_data_6d(bvh_filename):
    print(f'---------------------------get_training_format_data_6d: raw_data----------')
    data = parse_frames(bvh_filename)

    all_frames_data = []

    
    for frame in data:
        hip_frame = frame[:3] * 0.01
        euler_angles = np.array(frame[3:] * np.pi / 180).reshape(-1, 3)
        joint_frame = extra.euler_angles_to_matrix(torch.Tensor(euler_angles), convention='ZXY')
        rotation_matrix = extra.matrix_to_rotation_6d(torch.Tensor(joint_frame)).numpy()
        # rotation_matrix = np.concatenate(rotation_matrix)

        new_data = list(hip_frame) + rotation_matrix.tolist()

        # print("new_data length:", len(new_data))
        
        all_frames_data.append(new_data)
        # print("all_frames_data length:", len(all_frames_data))
    
    return np.array(all_frames_data)

import numpy as np
import torch

def get_training_data_quaternions(bvh_file):
    """
    Convert rotation matrices from BVH file to quaternions.
    Args:
        bvh_file: Path to the BVH file.
    Returns:
        List of quaternions.
    """
    print(f'---------------------------get_training_format_data_quaternions------------------')
    data = parse_frames(bvh_file)
    all_frames_data = []

    for frame in data:
        hip_frame = frame[:3] * 0.01
        euler_angles = np.array(frame[3:] * np.pi / 180).reshape(-1, 3)
        
        eul_mat = extra.euler_angles_to_matrix(torch.Tensor(euler_angles), convention='ZXY')
        
        quaternions = extra.matrix_to_quaternion(torch.Tensor(eul_mat)).numpy()
        # flaten the quaternions
        quaternions = quaternions.reshape(-1)
        new_data = list(hip_frame) + list(quaternions)
        # remove duble list from new_data
        # new_data = [item for item in new_data]


        all_frames_data.append(new_data)

        # all_frames_data = all_frames_data[0]
        # Assuming the lists are quaternion components

    # print("all_frames_data length:", np.array(all_frames_data).shape) # (5203, 175)
    # print("all_frames_data:", all_frames_data[:1])

    return np.array(all_frames_data)





def write_frames(format_filename, out_filename, data):
    
    format_lines = get_frame_format_string(format_filename)

    
    num_frames = data.shape[0]
    format_lines[len(format_lines)-2]="Frames:\t"+str(num_frames)+"\n"
    
    bvh_file = open(out_filename, "w")
    bvh_file.writelines(format_lines)
    bvh_data_str=vectors2string(data)
    bvh_file.write(bvh_data_str)    
    bvh_file.close()

def regularize_angle(a):
	
	if abs(a) > 180:
		remainder = a%180
		print ('hi')
	else: 
		return a
	
	new_ang = -(sign(a)*180 - remainder)
	
	return new_ang

def write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, output_filename):
    bvh_vec_length = len(non_end_bones)*3 + 6
    
    out_data = np.zeros([len(xyz_motion), bvh_vec_length])
    for i in range(1, len(xyz_motion)):
        positions = xyz_motion[i]
        rotation_matrices, rotation_angles = helper.xyz_to_rotations_debug(skeleton, positions)
        new_motion1 = helper.rotation_dic_to_vec(rotation_angles, non_end_bones, positions)
								
        new_motion = np.array([round(a,6) for a in new_motion1])
        new_motion[0:3] = new_motion1[0:3]
								
        out_data[i,:] = np.transpose(new_motion[:,np.newaxis])
        
    
    write_frames(format_filename, output_filename, out_data)

def write_traindata_to_bvh(bvh_filename, train_data):
    seq_length=train_data.shape[0]
    # print(f'train_data.shape: {train_data.shape}')
    # print(f'train_data: {train_data}')
    # check for any nan values
    # print(f'np.isnan(train_data).any(): {np.isnan(train_data).any()}') #false
    # check for any inf values
    # print(f'np.isinf(train_data).any(): {np.isinf(train_data).any()}') #false

    # issues with the spesific joint
    # print(f'train_data[:, 0]: {train_data[:, 0]}')
    xyz_motion = []
    format_filename = standard_bvh_file
    # print(f'seq_length: {seq_length}')
    for i in range(seq_length):
        data = train_data[i]
        # print(f'data.shape: {data.shape}')  
        # print(f'data = train_data[i]: {data}')
        
        # print(f'a: {data}')
        data = np.array([round(a,6) for a in train_data[i]])
        #print data
        # print ('in write_traindata_to_bvh')
        # print(f'data. shape: {data.shape}')  #(174,)
        #print data
        # print(f'data: {data}')

        #input(' ' )
        position = data_vec_to_position_dic(data, skeleton)        
        
        
        xyz_motion.append(position)

        
    write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, bvh_filename)

def write_euler_traindata_to_bvh(out_filename, euler_data):

    # print(f'euler_data.shape: {euler_data.shape}')
    format_filename = standard_bvh_file
    # print(f'euler_data[:, :3]: {euler_data[:, :3]}')
    # De-normalize the hip
    euler_data[:, :3] /= 0.01

    # Convert back from radians to degrees
    bvh_data = []
    for frame in euler_data:
        frame[3:] = frame[3:] * 180 / np.pi  # Convert to degrees
        bvh_data.append(frame)
    
    # Use the write_frames function to output the data to a BVH file
    write_frames(format_filename, out_filename, np.array(bvh_data))

    
# def write_6d_traindata_to_bvh(bvh_filename, train_data):
#     seq_length=train_data.shape[0]

#     xyz_motion = []
#     format_filename = standard_bvh_file

#     for i in range(seq_length):
#         data = train_data[i]
#         print(f'data.shape: {data.shape}')
#         print(f'data: {data}')

#         data = np.array([[round(x, 6) for x in a] for a in train_data[i]])
#         position = data_vec_to_position_dic(data, skeleton)        
#         xyz_motion.append(position)
#         break
#     write_xyz_to_bvh(xyz_motion, skeleton, non_end_bones, format_filename, bvh_filename)






def write_6d_traindata_to_bvh(out_filename, train_data):
    print("-----------NEWWW6d to bvh-----------------")
    # print(f'type train data: {type(train_data)}')
    format_filename = standard_bvh_file
    num_frames = len(train_data)
    num_joints = (len(train_data[0]) - 3) // 6  # Calculate number of joints from the first frame

    all_frames_data = []

    for i in range(num_frames):
        # Separate the hip frame from the rest of the data
        hip_frame = train_data[i, :3] *180/np.pi # De-normalize the hip

        # Reshape the remaining data to the original format used for the conversion to 6D rotations
        rotation_6d = train_data[i, 3:].reshape(num_joints, 6)

        # Convert the 6D rotations back to rotation matrices
        joint_frame = extra.rotation_6d_to_matrix(torch.Tensor(rotation_6d))

        # Convert the rotation matrices back to Euler angles
        euler_angles = extra.matrix_to_euler_angles(joint_frame, convention='ZXY').numpy().flatten()

        # Combine the hip frame and Euler angles back to their original format
        frame_data = np.concatenate([hip_frame, euler_angles * 180 / np.pi])

        all_frames_data.append(frame_data)

    # Write the data back to a BVH file
    write_frames(format_filename, out_filename, np.array(all_frames_data))


def write_quaternion_traindata_to_bvh(out_filename, quaternions_data):
    print("-----------Quartenions to bvh-----------------")
    
    format_filename = standard_bvh_file
    all_frames_data = []

    for frame in quaternions_data:
        hip_frame = frame[:3] *100  # hip_frame is the first 3 elements
        quaternions = frame[3:].reshape(-1, 4)  # Reshape to have one quaternion per row

        euler_frames = []
        for quaternion in quaternions:
            rot_mat = extra.quaternion_to_matrix(torch.Tensor(quaternion)) 
            euler_angles = extra.matrix_to_euler_angles(rot_mat, convention='ZXY') * 180 / np.pi 

            euler_frames.append(euler_angles)

        # euler_frames = np.array(euler_frames).flatten()  # Flatten the list of lists
        euler_frames_flat = np.concatenate(euler_frames)  # Flatten euler frames
        new_data = np.concatenate([hip_frame, euler_frames_flat])  # Concatenate with hip_frame


        # print(f'type hip_frame: {type(hip_frame)}')
        # print(f'type euler_frames: {type(euler_frames)}')
        # print(f'hip_frame: {hip_frame}')
        # print(f'euler_frames: {euler_frames}')

        # new_data = list(hip_frame) + euler_frames  # Step 4

        all_frames_data.append(new_data)

    write_frames(format_filename, out_filename, np.array(all_frames_data))  # Added dtype=object here




# def write_6d_traindata_to_bvh(bvh_filename, train_data_hip, train_data_joint):
#     # Convert 6D rotations back to rotation matrices
#     train_data_hip_mat = extra.rotation_6d_to_matrix(torch.tensor(train_data_hip))
#     train_data_joint_mat = extra.rotation_6d_to_matrix(torch.tensor(train_data_joint))

#     # Compute the inverse of the hip rotation matrix
#     train_data_hip_mat_inv = train_data_hip_mat.inverse()

#     # Multiply the joint rotation matrix by the inverse of the hip rotation matrix
#     train_data_joint_mat = torch.bmm(train_data_joint_mat, train_data_hip_mat_inv)

#     # Convert rotation matrices to Euler angles
#     euler_angles_hip = extra.matrix_to_euler_angles(train_data_hip_mat, 'ZYX')
#     euler_angles_joint = extra.matrix_to_euler_angles(train_data_joint_mat, 'ZYX')

#     # Concatenate the hip and joint euler angles
#     euler_angles = torch.cat((euler_angles_hip, euler_angles_joint), dim=1)

#     # Write Euler angles to BVH file
#     write_frames(standard_bvh_file, bvh_filename, euler_angles)



    # def write_euler_traindata_to_bvh(bvh_filename, train_data):
    #     """
    #     This function converts euler angle training data back to a BVH file.
    #     Args:
    #         bvh_filename: The output path for the BVH file.
    #         train_data: The training data in euler angles as a numpy array of shape (num_frames, num_joints*3).
    #     """

    # # Convert numpy array to torch tensor
    # train_data = torch.from_numpy(train_data)

    # # skeleton, motion = bvh.read_bvh(bvh_filename)
    # # non_end_bones = get_non_end_bones(skeleton)

    # rotation_matrices = extra.euler_angles_to_matrix(train_data, 'YXZ')
    # # Convert rotation matrices to XYZ positions
    # xyz_positions = extra.rotation_matrices_to_xyz(rotation_matrices, skeleton, non_end_bones)

    # # Write the XYZ positions to a new BVH file
    # write_xyz_to_bvh(xyz_positions, skeleton, non_end_bones, bvh_filename, 'output.bvh')



       
        



    # for converting back the 6d to bvh
import torch
import numpy as np

# def write_6d_traindata_to_bvh(bvh_filename, train_data_6d):
#     seq_length = train_data_6d.shape[0]
#     # print(f'train_data_6d.shape: {train_data_6d.shape}') #(764, 342)
#     # print(f'train_data_6d: {train_data_6d}')
#     combined_data = []

#     for i in range(seq_length):
#         # print(f'train_data_6d[i].shape: {train_data_6d[i].shape}') 
#         data_6d = train_data_6d[i].reshape(-1,6) # Reshape the data to have the correct format
#         # print(f'data_6d.shape: {data_6d.shape}')
#         # print(f'data_6d: {data_6d}')
#         rotation_matrices = extra.rotation_6d_to_matrix(torch.tensor(data_6d))
#         # print(f'rotation_matrices: {rotation_matrices.shape}')
#         # print(f'rotation_matrices: {rotation_matrices}')

#         eulerAngles = [rotationMatrixToEulerAngles(rot_matrix.numpy()) for rot_matrix in rotation_matrices]
#         # print(f'eulerAngles: {eulerAngles}')
#         # print(f'eluerAngles.shape: {np.array(eulerAngles).shape}')
     
#         flatten_eul_angles = [item for sublist in eulerAngles for item in sublist] #this isnot used
#         # print(f'flatten_eul_angles.shape: {np.array(flatten_eul_angles).shape}') # (171,)
#         # print(f'flatten_eul_angles: {flatten_eul_angles}')
   
#         combined_data.append(eulerAngles)
#         # combined_data list of lists 
#         # combined_data = np.concatenate(combined_data)
#         # print(f'combined_data: {combined_data}')
#         # print(f'LEN(combined_data): {len(combined_data)}')
#         # print(f'train_data_6d[i, :3]: {train_data_6d[i, :3]}')
#         # print(f'len (train_data_6d[i, :3]): {len(train_data_6d[i, :3])}')

#     # print(f'combined_data.shape: {np.array(combined_data).shape}') #(764, 57, 3)
#     # wr6d_to_bvh(bvh_filename, np.array(combined_data))
#     write_frames(standard_bvh_file, bvh_filename, np.array(combined_data))






# def write_quaternion_traindata_to_bvh(bvh_filename, train_data):


    # convert 6d representation back to euler angles
# def write_6d_traindata_to_bvh(bvh_filename, train_data):
#     # Convert 6D npy data to Euler angles
#     eulerAngles_data = rotation_6d_to_euler_angles(train_data, non_end_bones)

#     # Convert Euler angles data to a list of positions
#     positions_data = [data_vec_to_position_dic(eulerAngles_data[i * 3:i * 3 + 3], skeleton) for i in range(len(non_end_bones))]

#     # Generate BVH file using the write_traindata_to_bvh function
#     write_traindata_to_bvh(bvh_filename, positions_data)


# # import torch
# def rotation_6d_to_euler_angles(rot6d_data, non_end_bones):
#     eulerAngles_data = np.zeros(len(non_end_bones) * 3)
#     for i, joint in enumerate(non_end_bones):
#         rot_matrix = extra.rotation_6d_to_matrix(torch.Tensor(rot6d_data[i * 6:i * 6 + 6]).unsqueeze(0)).numpy()[0]
#         eulerAngles_data[i * 3:i * 3 + 3] = rotationMatrixToEulerAngles(rot_matrix)
    
#     return eulerAngles_data

# def rotation_6d_to_euler_angles(rot6d_data, non_end_bones):
#     eulerAngles_data = np.zeros(len(non_end_bones) * 3)
#     for i, joint in enumerate(non_end_bones):
#         input_tensor = torch.Tensor(rot6d_data[i * 6:i * 6 + 6]).view(1, -1)  # Reshape the input tensor
#         print(f"Input tensor shape: {input_tensor.shape}")
#         rot_matrix = extra.rotation_6d_to_matrix(input_tensor).numpy()[0]
#         eulerAngles_data[i * 3:i * 3 + 3] = rotationMatrixToEulerAngles(rot_matrix)
    
#     return eulerAngles_data





# def data_vec_to_position_dic(data, skeleton):

#     expected_length = len(joint_index) * 3  # Each joint should have 3 coordinates (x, y, z)
    
#     # if len(data) != expected_length:
#     #     print(f"Warning: data length ({len(data)}) does not match the expected length ({expected_length})")

#     data = data * 100
#     hip_pos = data[joint_index['hip']*3:joint_index['hip']*3+3]
#     positions = {}

#     for joint in joint_index:
#         positions[joint] = data[joint_index[joint]*3:joint_index[joint]*3+3]

#     for joint in positions.keys():
#         if joint == 'hip':
#             positions[joint] = positions[joint]
#         else:
#             # Check if the joint position array is not empty
#             if positions[joint].shape[0] == 3:

#                 # print(f'point {joint} is {positions[joint]}')
#                 # print(f'point {joint} shape is {positions[joint].shape}')
#                 # print(f'hip pos is {hip_pos}')
#                 # print(f'hip shape is {hip_pos.shape}')
#                 positions[joint] = positions[joint] + hip_pos
#             else:
#                 print(f"Warning: Joint {joint} has an empty position array")

#     return positions

def data_vec_to_position_dic(data, skeleton):
    expected_length = len(joint_index) * 3  # Each joint should have 3 coordinates (x, y, z)
    
    if len(data) != expected_length:
        print(f"Warning: data length ({len(data)}) does not match the expected length ({expected_length})")
    data = data*100
    hip_pos=data[joint_index['hip']*3:joint_index['hip']*3+3]
    positions={}
    # for joint in joint_index:
    #     positions[joint]=data[joint_index[joint]*3:joint_index[joint]*3+3]
    for joint in joint_index:
        start_index = joint_index[joint]*3
        end_index = joint_index[joint]*3+3
        # print(f'Joint: {joint}, Start index: {start_index}, End index: {end_index}')  # Debug line
        positions[joint] = data[start_index:end_index]

    # if representation == 'positional':
    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            # Check if the joint position array is not empty
            # print(f'joint: {joint}')
            # print(f'positions[joint]: {positions[joint]}')            
            positions[joint]=positions[joint] + hip_pos
            
    return positions


       
def get_pos_dic(frame, joint_index):
    positions={}
    for key in joint_index.keys():
        positions[key]=frame[joint_index[key]*3:joint_index[key]*3+3]
    return positions

# Rotational_joint = len(read_bvh. rotational_joints_index)
# #split positional hp data from rotational to apply different losses
# def getHipAndRotationsFromSequence(batch_sequence, joint_size, frame_hip_index, sequence_length):
#     positions = torch.zeros((batch_sequence.shape[0], sequence_length*3)) 
#     rotations = torch.zeros((batch_sequence.shape[0], batch_sequence.shape[1]))
#     for b in range (batch_sequence.shape[0]):
#         for s in range (sequence_length):
#             sequence_shift = (joint_size*Rotational_joint + 3) * s
#             sequence_shift_pos = 3 * s
#             =
#             -
#             sequence_length*3))

#             batch_sequence [b, sequence_shift+frame_            sequence_shift_rot = (joint_size*Rotational_joint) * s positions [b][sequence_shift_pos: sequence_shift_pos+3] rotations [b][sequence_shift_rot: sequence_shift_rot+ (Rotational_joint*joint_size)] = torch.cat([ batch_sequence [b, sequence_shift: sequence_shift+frame_hip_index*joint_size], batch_sequence [b, sequence_shift+frame_hip_index*joint_size +3:3+sequence_shift+ ((Rotationa
#             ], dim=0)
#     return rotations, positions

#######################################################
#################### Write train_data to bvh###########                



def vector2string(data):
    s=' '.join(map(str, data))
    
    return s

def vectors2string(data):
    s='\n'.join(map(vector2string, data))
   
    return s
 
    
def get_child_list(skeleton,joint):
    child=[]
    for j in skeleton:
        parent=skeleton[j]['parent']
        if(parent==joint):
            child.append(j)
    return child
    
def get_norm(v):
    return np.sqrt( v[0]*v[0]+v[1]*v[1]+v[2]*v[2] )

def get_regularized_positions(positions):
    
    org_positions=positions
    new_positions=regularize_bones(org_positions, positions, skeleton, 'hip')
    return new_positions

def regularize_bones(original_positions, new_positions, skeleton, joint):
    children=get_child_list(skeleton, joint)
    for child in children:
        offsets=skeleton[child]['offsets']
        length=get_norm(offsets)
        direction=original_positions[child]-original_positions[joint]
        #print child
        new_vector=direction*length/get_norm(direction)
        #print child
        #print length, get_norm(direction)
        #print new_positions[child]
        new_positions[child]=new_positions[joint]+new_vector
        #print new_positions[child]
        new_positions=regularize_bones(original_positions,new_positions,skeleton,child)
    return new_positions

def get_regularized_train_data(one_frame_train_data):
    
    one_frame_train_data=one_frame_train_data*100.0
    positions={}
    for joint in joint_index:
        positions[joint]=one_frame_train_data[joint_index[joint]*3:joint_index[joint]*3+3]
    
    #print joint_index['hip']
    hip_pos=one_frame_train_data[joint_index['hip']*3:joint_index['hip']*3+3]

    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
            
    
    new_pos=get_regularized_positions(positions)
    
    
    new_data=np.zeros(one_frame_train_data.shape)
    i=0
    for joint in new_pos.keys():
        if (joint!='hip'):
            new_data[i*3:i*3+3]=new_pos[joint]-new_pos['hip']
        else:
            new_data[i*3:i*3+3]=new_pos[joint]
        i=i+1
    new_data=new_data*0.01
    return new_data

def check_length(one_frame_train_data):
    one_frame_train_data=one_frame_train_data*100.0
    positions={}
    for joint in joint_index:
        positions[joint]=one_frame_train_data[joint_index[joint]*3:joint_index[joint]*3+3]
    
    #print joint_index['hip']
    hip_pos=one_frame_train_data[joint_index['hip']*3:joint_index['hip']*3+3]

    for joint in positions.keys():
        if(joint == 'hip'):
            positions[joint]=positions[joint]
        else:
            positions[joint]=positions[joint] +hip_pos
    
    for joint in positions.keys():
        if(skeleton[joint]['parent']!=None):
            p1=positions[joint]
            p2=positions[skeleton[joint]['parent']]
            b=p2-p1
            #print get_norm(b), get_norm(skeleton[joint]['offsets'])
    
    


		























