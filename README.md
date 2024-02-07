# NeSS: Neural SDF in Structured Light Systems

The PyTorch implementation of the 3DV 2024 paper: Depth Reconstruction with Neural Signed Distance Fields in Structured Light Systems.

if you find our code useful, please cite:

> @inproceedings{qiao2024depth,
>   title={Depth Reconstruction with Neural Signed Distance Fields in Structured Light Systems},
>   booktitle={International Conference on 3D Vision},
>   year={2024}
> }


## Step 0: Installation

The project is implemented based on PyTorch and OpenCV.

You can install the required packages by:

```
pip install -r requirements.txt
```

TODO: We wrote a small library called 'pointerlib' for swift developing. In the final version we will remove the reliance of this small library. For now, you still need to install the wheel file from [here](https://drive.google.com/file/d/1LqAAnYo1i-xe1Yza_p1BIU9HEEt6DDLa/view?usp=sharing) and install that by `pip`.

Our code is using `python==3.8` and `torch==1.13.1`.

P.S: `Open3D` and `openpyxl` are only used for visualization in folder `./tools`. You can ignore these two if you only run the main project.


## Step 1: Prepare your datasets

### Download the datasets

Please download the dataset from [here](https://drive.google.com/file/d/1pnIXH4n1XVz1vCrzfvQcuuxyjCs_5Xcq/view?usp=sharing). The calibration parameters are stored in `config.ini`.

Remember to update the dataset path for scripts under `./scripts` folder.

For example, if you save the dataset under the path `/home/codepointer/dataset/NeSSDataset`, then you should modify the `DATA_DIR` variables for scripts `./scripts/*.sh`:

```
DATA_DIR="/home/codepointer/dataset/NeSSDataset"
```

### (Optional) Prepare your own Dataset

TODO

## Step 2: Train the model

You can start to run the code by:

```
chmod +x ./scripts/run-main.sh
./scripts/run-main.sh {YourGPUDeviceNumber}
```

Please replace the above `{YourGPUDeviceNumber}` during your training.

## (Optional) Step 3: Visualize you results

We output the `.ply` file and `.png` files under the `output` folder. You can also check the logs by using tensorboard.

We also provide several tools for better analyzing under the folder `./tools`.

