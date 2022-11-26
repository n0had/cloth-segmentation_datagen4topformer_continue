import os
import sys
import time
import yaml
import cv2
import pprint
import traceback
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from data.custom_dataset_data_loader import CustomDatasetDataLoader, sample_data


from options.base_options import parser
from utils.tensorboard_utils import board_add_images
from utils.saving_utils import save_checkpoints
from utils.saving_utils import load_checkpoint, load_checkpoint_mgpu
from utils.distributed import get_world_size, set_seed, synchronize, cleanup

from networks import U2NET


def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

    # Saving options in yml file
    option_dict = vars(opt)
    with open(os.path.join(opt.save_dir, "training_options.yml"), "w") as outfile:
        yaml.dump(option_dict, outfile)

    for key, value in option_dict.items():
        print(key, value)


def training_loop(opt):

    custom_dataloader = CustomDatasetDataLoader()
    custom_dataloader.initialize(opt)
    loader = custom_dataloader.get_loader()
    
    custom_dataloader.printTensorSizes(10)
    custom_dataloader.printTensorSizes(27)
    savePath = "/content/drive/MyDrive/tapmobileTestProj2/mmseg_clothes_datagen/try1"
    custom_dataloader.callSaveImagePair(10, savePath, True, 1)
    custom_dataloader.callSaveImagePair(27, savePath, False, 2)
    

if __name__ == "__main__":

    opt = parser()

    if opt.distributed:
        if int(os.environ.get("LOCAL_RANK")) == 0:
            options_printing_saving(opt)
    else:
        options_printing_saving(opt)

    try:
        if opt.distributed:
            print("Initialize Process Group...")
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

        set_seed(1000)
        training_loop(opt)
        cleanup(opt.distributed)
        print("Exiting..............")

    except KeyboardInterrupt:
        cleanup(opt.distributed)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        cleanup(opt.distributed)
