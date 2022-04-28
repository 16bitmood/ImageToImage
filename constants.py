import torch
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
LOAD_CHECKPOINT = False
SAVE_CHECKPOINT = True

NUM_EPOCHS  = 200
BATCH_SIZE  = 1

LEARNING_RATE = 0.0002
L1_LAMBDA   = 100
L1_LAMBDA_IDENTITY = 0
BETAS       = (0.5, 0.999)

BASE_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])