import torch
import dataset

SEED=45
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = ["cifar10", "ffhq_96", "ffhq_128"]
SAMPLING_TYPES = ["ODE", "SDE"]
SDE_SAMPLING_MODES = ["langevin_mcmc_and_euler_maruyama", "euler_maruyama_only"]

DATASET_NAME = DATASETS[0]
BATCH_SIZE = 64
IMAGE_SIZE, CHANNELS, DIMENSIONS, EMBEDDING_SIZE = dataset.get_dimensions(DATASET_NAME)

GROUPS_NUMBER = 32
SAMPLING_TYPE = SAMPLING_TYPES[1] # "SDE"
SDE_SAMPLING_MODE = SDE_SAMPLING_MODES[0] # "langevin_mcmc_and_euler_maruyama"
ODE_ERROR_TOLERANCE = 1e-5
EPSILON = 1e-5
SIGMA = 25.0
SCALE = 30.0
SIGNAL_TO_NOISE_RATIO = 0.16

LEARNING_RATE = 4e-4
EPOCHS = 10_000
T = 4_000

CHECKPOINT_FREQUENCY = 10
CHECKPOINT_FILE = None # Otherwise, for example: "checpoints/cifar10_checkpoint_epoch_100.pt"
