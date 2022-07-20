
import numpy as np
import torch
from torch._utils import classproperty
import logging
import logging.config
from os import path

from IPython.terminal.interactiveshell import TerminalInteractiveShell

import random
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

from augmentations import*
#from utilities import *

import cv2

from resnet_unet import*
import numpy as np
import os, types 
import pandas as pd 
#from botocore.client import Config 

from albumentations import Compose, HorizontalFlip, Normalize, PadIfNeeded, Resize
from itkwidgets import view


import io
from utilities import*

from data2 import*
from data33 import*

from ignite.contrib.handlers import CosineAnnealingScheduler,LinearCyclicalScheduler,ConcatScheduler
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from toolz import compose
from torch.utils import data
from batch import*
from utils import *
import  logging_handlers


from metrics import*
#from dutchf3utils import*

from tensorboard_handlers import*




from engine import*
from metrics import *
from itkwidgets import view


from torch.utils import data



import yacs.config

def get_vis():


    CONFIG_FILE = (
        "seresnet_unet.yaml"
    )

    # number of images to score
    N_EVALUATE = 1
    # demo flag - by default notebook runs in demo mode and only fine-tunes the pre-trained model. Set to False for full re-training.
    DEMO = False
    # options are test1 or test2 - picks which Dutch F3 test set split to use
    TEST_SPLIT = "test1"



    import os

    assert isinstance(N_EVALUATE, int) and N_EVALUATE>0, "Number of images to score has to be a positive integer"
    assert isinstance(DEMO, bool), "demo mode should be a boolean"
    assert TEST_SPLIT == "test1" or TEST_SPLIT == "test2"

    with open(CONFIG_FILE, "rt") as f_read:
        config = yacs.config.load_cfg(f_read)

    #!python test_sesmic.py

    max_snapshots = config.TRAIN.SNAPSHOTS
    papermill = False

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    # Fix random seeds:
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(seed=config.SEED)

    # load a model
    #torch.cuda.set_enabled_lms(True)
    model = get_seg_model(config)#model_f3_nb_resnet_unet_11124.pt

    # #model_f3_nb_resnet_unet_14832
    # #model_f3_nb_resnet_unet_14832.pt
    # #model_f3_nb_resnet_unet_14832.pt
    # #model_f3_nb_resnet_unet_24102.pt#model_f3_nb_resnet_unet_24102.pt
    # #model_f3_nb_resnet_unet_12978.pt

    # trained_model = torch.load("model_f3_nb_resnet_unet_12978.pt")

    # #

    # trained_model = {k.replace("module.", ""): v for (k, v) in trained_model.items()}
    # model.load_state_dict(trained_model, strict=True)

    # Your data file was loaded into a botocore.response.StreamingBody object.
    # Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
    # ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
    # pandas documentation: http://pandas.pydata.org/


    # Send to GPU if available

    # Send to GPU if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # model = model.to(device)

    # # SGD optimizer
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config.TRAIN.MAX_LR,
    #     momentum=config.TRAIN.MOMENTUM,
    #     weight_decay=config.TRAIN.WEIGHT_DECAY,
    # )

    # # learning rate scheduler
    # scheduler = CosineAnnealingScheduler(
    #     optimizer, "lr", config.TRAIN.MAX_LR, config.TRAIN.MIN_LR, cycle_size=snapshot_duration
    # )

    # # weights are inversely proportional to the frequency of the classes in the training set
    # class_weights = torch.tensor(
    #     config.DATASET.CLASS_WEIGHTS, device=device, requires_grad=False
    # )

    # # loss function
    # criterion = torch.nn.CrossEntropyLoss(
    #     weight=class_weights, ignore_index=255, reduction="mean"
    # )

    shell:TerminalInteractiveShell = TerminalInteractiveShell.instance()

    # Commented out IPython magic to ensure Python compatibility.
    if not papermill:
    #     %load_ext tensorboard
        shell.run_line_magic("load_ext","tensorboard")

    if not papermill:
    #     %tensorboard --logdir $output_dir --port 9001 --host 0.0.0.0
        shell.run_line_magic("tensorboard","--logdir $output_dir --port 9001 --host 0.0.0.0")

    trained_model = torch.load("model_f3_nb_resnet_unet_25956_300epoch.pt",map_location=torch.device('cpu'))
    trained_model = {k.replace("module.", ""): v for (k, v) in trained_model.items()}
    model.load_state_dict(trained_model, strict=True)
    model = model.to(device)

    # Augmentation
    # augment entire sections with the same normalization
    from torch.utils import data
    section_aug = Compose(
        [Normalize(mean=(config.TRAIN.MEAN,), std=(config.TRAIN.STD,), max_pixel_value=1,)]
    )

    # augment each patch and not the entire sectiom which the patches are taken from
    patch_aug = Compose(
        [
            Resize(
                config.TRAIN.AUGMENTATIONS.RESIZE.HEIGHT,
                config.TRAIN.AUGMENTATIONS.RESIZE.WIDTH,
                always_apply=True,
            ),
            PadIfNeeded(
                min_height=config.TRAIN.AUGMENTATIONS.PAD.HEIGHT,
                min_width=config.TRAIN.AUGMENTATIONS.PAD.WIDTH,
                border_mode=config.OPENCV_BORDER_CONSTANT,
                always_apply=True,
                #mask_value=255,
                
            ),
        ]
    )

    # Process test data
    pre_processing = compose_processing_pipeline(config.TRAIN.DEPTH, aug=patch_aug)
    output_processing = output_processing_pipeline(config)

    # Select the test split
    split = TEST_SPLIT

    #labels = np.load("/content/test1_labels.npy")
    # section_file = path.join(config.DATASET.ROOT, "splits", "section_" + split + ".txt")
    # write_section_file(labels, section_file, config)

    # Load test data
    TestSectionLoader = get_test_loader(config)
    test_set = TestSectionLoader(
        config, split=split, is_transform=True, augmentations=section_aug
    )
    # needed to fix this bug in pytorch https://github.com/pytorch/pytorch/issues/973
    # one of the workers will quit prematurely
    torch.multiprocessing.set_sharing_strategy("file_system")
    test_loader = data.DataLoader(
        test_set, batch_size=1, num_workers=config.WORKERS, shuffle=False
    )

    CLASS_NAMES = [
        "upper_ns",
        "middle_ns",
        "lower_ns",
        "rijnland_chalk",
        "scruff",
        "zechstein",
    ]

    n_classes = len(CLASS_NAMES)

    # keep only N_EVALUATE sections to score
    test_subset = random.sample(list(test_loader), 1)

    results = list()
    running_metrics_split = runningScore(n_classes)

    # testing mode
    with torch.no_grad():
        model.eval()
        # loop over testing data
        for i, (images, labels) in enumerate(test_subset):
            logger.info(f"split: {split}, section: {i}")
            outputs = patch_label_2d(
                model,
                images,
                pre_processing,
                output_processing,
                config.TRAIN.PATCH_SIZE,
                config.TEST.TEST_STRIDE,
                config.VALIDATION.BATCH_SIZE_PER_GPU,
                device,
                n_classes,
            ) 

            pred = outputs.detach().max(1)[1].numpy()
            gt = labels.numpy()
            
            # update evaluation metrics
            running_metrics_split.update(gt, pred)
            
            # keep ground truth and result for plotting
            results.append((np.squeeze(gt), np.squeeze(pred)))



    results[0][0][200]

    fig = plt.figure(figsize=(15, 50))
    # only plot a few images
    nplot = min(N_EVALUATE, 10)
    for idx in range(nplot):
        # plot actual
        plt.subplot(nplot, 2, 2 * (idx + 1) - 1)
        plt.imshow(results[idx][0])
        # plot predicted
        plt.subplot(nplot, 2, 2 * (idx + 1))
        plt.imshow(results[idx][1])
        
    f_axes = fig.axes
    _ = f_axes[0].set_title("Actual")
    _ = f_axes[1].set_title("Predicted")

    path_to_save_img = os.path.join(os.path.dirname(os.path.realpath(__file__)),'static/images/img.png')

    plt.savefig(path_to_save_img)
    print('Image Saved..')


    
