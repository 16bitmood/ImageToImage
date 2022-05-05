import torch
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
LOAD_CHECKPOINT = True
SAVE_CHECKPOINT = False

NUM_EPOCHS = 200
BATCH_SIZE = 1

LEARNING_RATE = 1e-5
L1_LAMBDA = 100
L1_LAMBDA_CYCLE = 10
L1_LAMBDA_IDENTITY = 0
BETAS = (0.5, 0.999)

BASE_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])