import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append('../')
from file_helper import *
import random
import cv2
import numpy as np

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from coco_loader import * 
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch
from pytorch_lightning import Trainer




class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay ,train_loader , test_loader):
        super().__init__()
        '''
        config = DetrConfig(use_pretrained_backbone=True,num_labels=1,)
        self.model = DetrForObjectDetection(config)
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            #num_labels=len(id2label),
            num_labels=2,
            #ignore_mismatched_sizes=True
            ignore_mismatched_sizes=False
        )
        '''
        self.model = DetrForObjectDetection(DetrConfig(num_labels=2
                                                      ,ignore_mismatched_sizes=True
                                                      ,backbone='resnet50'))
        #print("init model " , self.model)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        #loss_dict['loss_ce']*=20
        #print(loss_dict)

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        #return TRAIN_DATALOADER
        return self.train_loader

    def val_dataloader(self):
        #return VAL_DATALOADER
        return self.test_loader


if __name__ =="__main__":
    print(os.getcwd())    
    train_file = os.path.join(os.path.dirname(os.getcwd()) , "anno" , "coco_train_200.json")
    test_file = os.path.join(os.path.dirname(os.getcwd()) , "anno" , "coco_test_10.json")
    #train_file = "coco_train_200.json"
    #test_file = "coco_test_10.json"

    train_loader , test_loader , train_dataset , test_dataset = get_loader(
        img_root_folder="E:/Projects/Datasets/Zillow_coco/train_onedrive/" ,
        try_load=True,
        train_json_path= train_file , 
        test_json_path= test_file,
        batch_size=8
            )
    '''
    '''
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4 ,train_loader= train_loader , test_loader= test_loader)

    MAX_EPOCHS = 50
    trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=2)

    trainer.fit(model)
    
    # Save Model
    output_path = create_folder(os.path.join(os.getcwd(), "output" , "checkpoints"))
    MODEL_PATH = os.path.join( output_path, 'custom-model')
    model.model.save_pretrained(MODEL_PATH)
    pass