# depth estimation network

## Requirements
```
-Python 3.8+
-PyTorch >= 1.10 (CUDA-enabled for GPU training)
-torchvision
-numpy, scipy, pillow
-opencv-python
-cikit-image
-tqdm, matplotlib
```
## Dataset preparation
In the config.py, please set your clean image dataset path
```
Image_DIR = "datasets"
```
## Run Model
To run the depth estimation network:

1: Please download the checkpoints form: [download link](https://drive.google.com/file/d/1zGE0wVO2ma6YxnvxlGow7XUlZg5ITaVu/view?usp=sharing), then extract them into the checkpoints folder. 

2: Run the save_results.py

3: The model outputs is saved in the "result" folder. To change it, in the config.py, please set your output image path
```
OUTPUT_DIR = "results"
```
