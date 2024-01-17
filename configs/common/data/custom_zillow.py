import itertools
import os

import sys
sys.path.append('../')
from config import * # my config
#os.chdir(current_file)


from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()

register_coco_instances("my_dataset_train", {}, ZILLOW_TRAIN_COCO_JSON_PATH, ZILLOW_DATASET_FOLDER)
register_coco_instances("my_dataset_test", {}, ZILLOW_TEST_COCO_JSON_PATH,  ZILLOW_DATASET_FOLDER)

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="my_dataset_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=0,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="my_dataset_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=0,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


if __name__=="__main__":
	
	pass