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

from model import *

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