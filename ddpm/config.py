import torch
import dataset 

SEED=45
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = ["cifar10", "ffhq_96", "ffhq_128"]
DATASET_NAME = DATASETS[1]

BATCH_SIZE = 64
IMAGE_SIZE, CHANNELS = dataset.get_dimensions(DATASET_NAME)

BETA_INITIAL = 1e-4
BETA_FINAL = 1e-2

FEATURE_MAP_SIZE = 64
GROUPS_NUMBER = 32
HEADS_NUMBER = 1
BLOCKS_NUMBER = 1

LEARNING_RATE = 2e-5
EPOCHS = 1_000
T = 1_000

CHECKPOINT_FREQUENCY = 10
CHECKPOINT_FILE = None # Otherwise, for example: "checpoints/cifar10_checkpoint_epoch_100.pt"
