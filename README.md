# acLSTM-motion-prediction
This project is focused on the study of character motion prediction, particularly exploring various character motion data representations. The objective is to expand this area by testing more representations and evaluating their performance.

The study is divided into three main parts:

Preprocessing: The provided BVH files are converted into four distinct character motion data representations - Positional, Euler Angles, 6D, and Quaternions.
Training: Training procedures and loss functions are tailored to each specific representation to optimize the learning process for the neural network.
Evaluation: Performance comparison of the network across different data representations.
This README provides instructions on how to run the scripts for preprocessing, training, and evaluation.

#### Setup
Clone this repository to your local machine.

Python 3.6 or higher is required to run the scripts. If you don't have Python installed, you can download it from [here](https://www.python.org/downloads/).

You will need  [Pytorch](https://pytorch.org/) as well.

You will also need to have transforms3d, which can be installed by using this command:
```
pip install transforms3d
```
## Preprocessing - Generation Data scripts:
The preprocessing stage involves converting BVH files into four distinct character motion data representations. The script provided transforms BVH files into training data representations including positional, Euler angle, 6D, and quaternion representations.


1. Go to the root directory of the project.
2. Open the "generate_training_data.py" script.
3. Uncomment the code for each representation you want to generate training data for, and make sure to specify the correct input and output directories for your BVH files and processed data.

*Note: The training data are already generated and saved inside training_data folder*


## Running Training Scripts
In the directory code/training exist 4 training python scripts you should run in order to train each representation data produced on the above step.

Run each training script from the root directory.

For each preprocessing script, you will need to update the following directory paths at the end of the file:

    ```python
    read_weight_path=""
    # Location to save the weights of the network during training
    write_weight_folder="./train_weight_aclstm_martial/"
    # Location to save the temporary output of the network and the groundtruth motion sequences in the form of BVH
    write_bvh_motion_folder="./train_tmp_bvh_aclstm_martial/"
    # Location of the training data
    dances_folder = "./train_data_xyz/martial/"
    ```

*Note: Already pretrained weights exist in the ./weights directory for each representation. If you want to load a weight, adjust the read_weight_path variable accordingly.*


## Evaluation

In the evaluation stage, you will analyze and compare the performance of the network across different data representations.
Running Evaluation Scripts

For qualitative result testing
Implement the evaluation script in the evaluation folder (divided into four scripts) for each representation and review the generated motion in the synthetic motion folder. Necessary alterations have to be made to the concluding lines to designate the "read_weight_path", referring to the storage location of the desired network weights for testing, and "write_bvh_motion_folder", indicating the destination for the output motions. Furthermore, "dances_folder" is where the code will randomly select a brief initial sequence. You might also wish to adjust the "batch" to control the number of motion clips to be generated, as well as "generate_frames_numbers" to set the duration of the motion clips among other parameters.
1.	Update the paths for saved weights and temporary outputs in the evaluation scripts.
2.	Run the evaluation script from the root folder
The resultant motions generated by the network frequently exhibit irregularities such as sliding feet and occasionally feet underneath the ground. If these artifacts are unsatisfactory, they can be rectified using fix_feet.py. 

In order to evaluate and compare the Mean Squared Error (MSE) results for each set of trained weights, execute the loss_evaluation script. This will transform all temporary BVH files into positional data, subsequently calculating the MSE for each pair.			
Rendering of the bvh motion can be accomplished using software like MotionBuilder, Maya, 3D max, or for the easiest approach, utilize an online BVH renderer, for example: http://lo-th.github.io/olympe/BVH_player.html

