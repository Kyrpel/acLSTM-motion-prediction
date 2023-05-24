# acLSTM-motion-prediction
This project is focused on the study of character motion prediction, particularly exploring various character motion data representations. The objective is to expand this area by testing more representations and evaluating their performance.

The study is divided into three main parts:

Preprocessing: The provided BVH files are converted into four distinct character motion data representations - Positional, Euler Angles, 6D, and Quaternions.
Training: Training procedures and loss functions are tailored to each specific representation to optimize the learning process for the neural network.
Evaluation: Performance comparison of the network across different data representations.
This README provides instructions on how to run the scripts for preprocessing, training, and evaluation.

## Setup
Clone this repository to your local machine.

Python 3.6 or higher is required to run the scripts. If you don't have Python installed, you can download it from [here](https://www.python.org/downloads/).

You will need  [Pytorch](https://pytorch.org/) as well.

You will also need to have transforms3d, which can be installed by using this command:
```
pip install transforms3d
```
## Preprocessing
The preprocessing stage involves converting BVH files into four distinct character motion data representations. The script provided transforms BVH files into training data representations including positional, Euler angle, 6D, and quaternion representations.

### Running Generation Data scripts:

1. Go to the root directory of the project.
2. Open the "generate_training_data.py" script.
3. Uncomment the code for each representation you want to generate training data for, and make sure to specify the correct input and output directories for your BVH files and processed data.
Note: The training data are already generated and saved inside training_data folder