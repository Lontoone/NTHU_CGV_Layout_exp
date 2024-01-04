import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
import torchvision
import sys
sys.path.append('../')
#sys.path.append('../file_helper.py')
from config import *
from tqdm import tqdm
from model import *
from file_helper import *


# DataLoader
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def uv_to_xyxy(u,v ):
    result = torch.zeros_like(v)
    result[:,0] = u[:,0] # top left x
    result[:,1] = v[:,0]
    result[:,2] = u[:,1] # right bottom x
    result[:,3] = v[:,3]
    return result

def debug_draw_bbox(img , bbox , normalize = True ):    
    h,w = img.shape[-2:]
    debug_img = img.permute(1,2,0).detach().cpu().numpy()
    debug_img = np.ascontiguousarray(debug_img)

    boxes = bbox.reshape(-1,4)
    for box in boxes:
        if(normalize):
            p0 = np.int32([box[0] * w , box[1] * h])
            p1 = np.int32([box[2] * w , box[3] * h])
        else:
            p0 = np.int32([box[0]  , box[1] ])
            p1 = np.int32([box[2]  , box[3] ])
        debug_img = cv2.rectangle(debug_img, tuple(p0) ,  tuple(p1)  , (0,255,0) , 2 )
    return debug_img

# Define your custom collate_fn
def collate_fn(data):
    # data is a list of tuples with (example, label)
    tensors, targets = zip(*data)
    '''
    # Pad the tensors and stack the targets
    tensors = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    '''
    
    return tensors, targets

class ZillowDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None , anno_path="" , mode ="train" , device = 'cuda'):        
        self.transforms = transforms
        self.mode= mode
        self.annos = []
        self.device = device

        with open(anno_path  , 'r') as f:
            self.annos =  json.load(f)
            print("anno loaded " , len(self.annos))        
        
        

    def __getitem__(self, idx):
        # TODO: Load your data here        
        img_path = os.path.join(ZILLOW_DATASET_FOLDER , self.annos[idx]['image'] )        
        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        h,w = img.shape[-2:]
        u = torch.as_tensor(self.annos[idx]['u']) * w
        v = torch.as_tensor(self.annos[idx]['sticks_v']) *h

        target = {}
        bboxes = uv_to_xyxy(  u , v)        
        target['boxes'] = bboxes.to(self.device)
        target['labels'] = torch.ones(len(self.annos[idx]['u'])).view(-1).to(torch.int64) .to(self.device)
        
        # [Debug: ]
        '''
        for box in bboxes:
            debug_img = debug_draw_bbox(img , box)
            plt.imshow(debug_img)
            plt.show()
        '''

        #return img, target
        return img.to(self.device) , target

    def __len__(self):        
        return len(self.annos)


def anno_to_list(anno_b):
    anno_list =[]
    for anno in anno_b:
        anno_list.append({"boxes": anno["boxes"] , "labels" : anno['labels']})
    return anno_list
def loss_fn(output):
    losses = sum(loss for loss in output.values())
    return losses
def inf_loss_fn(output):
    labels = []
    boxes = []
    for b_out in output:
        label = b_out['labels']
        box = b_out['boxes']

        labels.append(label)
        boxes.append(box)
    labels = torch.cat(labels)
    boxes = torch.cat(boxes)

@torch.no_grad()
def write_loss(output , type='train' , step=0 ):
    for key, value in output.items():
        writer.add_scalar(f"{type}/{key}" , value , step)
    pass

@torch.no_grad()
def inf_fn(model , data_loader , vis_data , epoch , log_folder= ""):    
    model.train()    
    
    cnt = 0
    '''
    '''
    for data in tqdm(data_loader):
        img_b , anno_b = data
        anno_b_list = anno_to_list(anno_b)        
        output = model(img_b , anno_b_list)
        write_loss(output , 'test' , epoch * len(data_loader) + cnt)           
        cnt+=1

    # visualize
    model.eval()        
    for i in tqdm(range(5)):
        img , anno      = vis_data[i]                        
        img_b_norm      = transform_norm(img).unsqueeze(0)
        
        output_b        = model(img_b_norm)
        
        for j , out in enumerate(output_b):
            box = out['boxes'].detach().cpu().numpy()                        
            debug_img = debug_draw_bbox(img, box , normalize=False)

            gt_box = anno['boxes'].detach().cpu().numpy()                    
            debug_img_gt =  debug_draw_bbox(img, gt_box , normalize=False)

            fig, axs = plt.subplots(1, 2 ,  figsize=(20, 5))                                     
            # Plot the images
            axs[0].imshow(debug_img)
            axs[0].axis("off")
            axs[1].imshow(debug_img_gt)
            axs[1].axis("off")
            
            plt.savefig(os.path.join(log_folder , f"vis_test_ep{epoch}-it{i}.jpg") , dpi = 150)
            plt.close(fig)
        pass
    pass

# Test Train
def train_fn(model,optimizer  , epoch , data_loader , log_folder=""):
    cnt = 0
    for data in tqdm(data_loader):
        model.train()
        img_b , anno_b = data
        anno_b_list = anno_to_list(anno_b)        
        output = model(img_b , anno_b_list)

        loss = loss_fn(output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()    

        write_loss(output , 'train' , epoch * len(data_loader) + cnt)
        cnt+=1
            #break

        # [ Test ]
        '''
        '''
    if (epoch % 5 ==0):
        with torch.no_grad():
            model.eval()
            output_b        = model(img_b)
            debug_cnt = 0
            for img , anno , out in zip(img_b , anno_b , output_b):        
                pred_box = out['boxes'].detach().cpu().numpy()                        
                debug_img = debug_draw_bbox(img, pred_box , normalize=False)
                
                gt_bbox = anno['boxes']        
                debug_img_gt = debug_draw_bbox(img, gt_bbox.detach().cpu().numpy() ,normalize=False)
                fig, axs = plt.subplots(1, 2 ,  figsize=(20, 5))                                     
                # Plot the images
                axs[0].imshow(debug_img)
                axs[0].axis("off")
                axs[1].imshow(debug_img_gt)
                axs[1].axis("off")
                
                plt.savefig(os.path.join(log_folder , f"vis_train_ep{epoch}-it{debug_cnt}.jpg") , dpi = 150)
                plt.close(fig)

                debug_cnt+=1
    #return model

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    print("current on",os.getcwd())

    transform = T.Compose([
        transforms.Resize((512, 1024)),
        T.ToTensor(),  # convert PIL image to PyTorch tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize image
    ])
    transform_no_norm = T.Compose([
        transforms.Resize((512, 1024)),    
        T.ToTensor(),  # convert PIL image to PyTorch tensor
    ])
    transform_norm= transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_scale= transforms.Resize((512, 1024))

    dataset_train = ZillowDataset(transforms=transform , anno_path= '../anno/train_visiable_10k_no_cross.json' )
    dataset_test = ZillowDataset(transforms=transform , anno_path= '../anno/test_visiable_200_no_cross.json' )
    dataset_test_no_norm = ZillowDataset(transforms=transform_no_norm , anno_path= '../anno/test_visiable_20_no_cross.json' )
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=NUMBER_WORKESRS  , collate_fn = collate_fn  )
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=NUMBER_WORKESRS  , collate_fn = collate_fn )
    data_loader_test_no_norm = torch.utils.data.DataLoader(dataset_test_no_norm, batch_size=2, shuffle=False, num_workers=0  , collate_fn = collate_fn )

    #=========================================
    #               Setting 
    #=========================================
    MAX_TRAIN_EPOCHES = 160
    RUN_NAME = 'train_10k_final-3-start_ep5'
    

    pt_path = os.path.join(os.getcwd() , "checkpoints","train_10k_final-2","ep5.pth")
    #model_2cls = torch.load(pt_path)
    model_2cls.load_state_dict(torch.load(pt_path))    
    model_2cls = model_2cls.to('cuda')
    optimizer = optim.Adam(model_2cls.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    writer = SummaryWriter(RUN_NAME)

    train_eppch = 0
    eval_eppch = 0
    log_folder = create_folder( os.path.join(os.getcwd() , "output" , RUN_NAME) )
    print("============================= HI ===========================")
    '''
    '''
    
    for epoch in range(1,MAX_TRAIN_EPOCHES+1):
        print("epoch" , epoch)    
        train_fn(model_2cls , optimizer, train_eppch , data_loader_train , log_folder)
        train_eppch+=1

        if(epoch % 5 ==0):
            inf_fn(model_2cls , data_loader_test , dataset_test_no_norm , eval_eppch , log_folder)
            eval_eppch+=1
        if(epoch % 5 ==0):
            save_path = create_folder (os.path.join(os.getcwd(),"checkpoints" , RUN_NAME))
            save_path = os.path.join(save_path , f'ep{epoch}.pth')
            #torch.save(model_2cls,save_path)
            torch.save(model_2cls.state_dict() ,save_path)
        else:
            save_path = create_folder (os.path.join(os.getcwd(),"checkpoints" , RUN_NAME))
            save_path = os.path.join(save_path , f'bk.pth')            
            torch.save(model_2cls.state_dict() ,save_path)
        #break
    scheduler.step()
    #inf_fn(model_2cls , data_loader_test , dataset_test_no_norm , 99 , log_folder)