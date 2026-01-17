import torch
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Image_DIR = "datasets"
OUTPUT_DIR = "results"
mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 1000
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "checkpoints/disc2.pth.tar"
CHECKPOINT_GEN = "checkpoints/gen.pth.tar"


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean, std) 
])
