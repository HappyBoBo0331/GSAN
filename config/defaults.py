from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# Changed by Xinchen Liu

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'baseline'
_C.MODEL.DIST_BACKEND = 'dp'
# Model backbone
_C.MODEL.BACKBONE = 'resnet50'
# Last stride for backbone
_C.MODEL.LAST_STRIDE = 1
# If use IBN block
_C.MODEL.WITH_IBN = False
# Global Context Block configuration
_C.MODEL.STAGE_WITH_GCB = (False, False, False, False)
_C.MODEL.GCB = CN()
_C.MODEL.GCB.ratio = 1./16.
# If use imagenet pretrain model
_C.MODEL.PRETRAIN = True
# Pretrain model path
_C.MODEL.PRETRAIN_PATH = '/home/liuxinchen3/notespace/project/resnet_pretrain/resnet50-19c8e357.pth'
# Checkpoint for continuing training
_C.MODEL.CHECKPOINT = ''
_C.MODEL.VERSION = ''
_C.MODEL.NUM_PARTS = 1

#
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 256]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 256]
# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random lightning and contrast change 
_C.INPUT.DO_LIGHTING = False
_C.INPUT.MAX_LIGHTING = 0.2
_C.INPUT.P_LIGHTING = 0.75
# Random erasing
_C.INPUT.DO_RE = True
_C.INPUT.RE_PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("market1501",)
# List of the dataset names for testing
_C.DATASETS.TEST_NAMES = ("market1501",)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.DIST = False

_C.SOLVER.OPT = "adam"

_C.SOLVER.LOSSTYPE = ("softmax",)

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.LOG_INTERVAL = 10
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.NORM = True
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"


# ---------------------------------------------------------------------------- #
# NCE options
# ---------------------------------------------------------------------------- #
_C.NCE = CN()
_C.NCE.K = 4096
_C.NCE.T = 0.07
_C.NCE.M = 0.5