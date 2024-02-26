# [DataLoader]
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from torch.utils.data import random_split
import json
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt 
import cv2
import numpy as np
from scipy.spatial import distance

import sys
sys.path.append('../')
print(os.getcwd())
from config import *
sys.path.append('../Horizon_and_SAM/Horizon/')
from horizon_utlis import *

IMG_SIZE =  np.array([IMG_WIDTH, IMG_HEIGHT]) # [w,h]

def collate_fn(batch):
    return tuple(zip(*batch))
#=================================
#             Augmentation
#=================================
def gauss_noise_tensor(img):
    if np.random.rand() < 0.5 and Horizon_AUG:
        sigma = np.random.rand() *0.125
        out = img + sigma * torch.randn_like(img)
        return out
    return img

def blank(img):    
    return img

class CustomDataset(Dataset):
    global debug_current_imgs_path
    def __init__(self, annotations_file_path , img_size=[IMG_WIDTH, IMG_HEIGHT] , use_aug= True , padding_count = 90 , c= 0.1):
        # Open json file        
        json_path =  annotations_file_path
        f= open(json_path)
        anno = json.loads(f.read())
        f.close()
        self.anno = anno
        self.img_size = img_size
        self.padding_count = padding_count
        self.c = c


        #do_jitter = np.random.rand() > 0.5 if Horizon_AUG else False        
        self.use_aug = use_aug
        
        self.transform = transforms.Compose([    
            transforms.ToPILImage(),                    
            transforms.Resize((img_size[1], img_size[0])),            
            #transforms.ColorJitter((0.4 , 1) , (0.7,1) , (0.6,1) , (-0.5, 0.5)) if self.use_aug else blank,        
            transforms.ToTensor(),            
            #gauss_noise_tensor if self.use_aug else blank,
        ])


    def __len__(self):
        return len(self.anno)
    def get_bbox_count(self ):
        count =0
        for data in self.anno:            
            count += len(data['bboxes'])
        return count

    def __getitem__(self, idx):      
        img_path = os.path.join( ZILLOW_DATASET_FOLDER, self.anno[idx]['image'])
        image = cv2.imread(img_path)          

        if(self.transform!=None):
            image= self.transform( image)

        if self.use_aug:
            do_flip = np.random.rand() > 0 
            do_roll = np.random.rand() > 0       
        else:   
            do_flip=False
            do_roll=False

        _, h,w = image.shape

        target = self.anno[idx]
        data = {}

        u = torch.tensor(target['u'])
        u_0idx = torch.where(u==0)[0]  #avoid u = 0 , may cause error while matching
        u[u_0idx,0] = 0.0001
        
        
        v = torch.tensor(target['sticks_v'])        
        du = u.flatten()[1::2] - u.flatten()[0::2]

        #============ Aug Transform ===========
        if do_flip:
            image = torch.flip(image, dims=[2])            
            u = torch.flip( 1 - u , [1])
            u_is_cross =  (u.flatten()[::2]<0).to(torch.float32)
            u_is_cross = u_is_cross.repeat_interleave(2)            
            u = (u.flatten() + 1* u_is_cross).reshape((-1,2))

            # 左右交換 => 0跟2互換 , 1跟3互換
            v_idx_all = torch.arange(v.numel())      
            v_idx = v_idx_all.clone()
            v_idx[0::4] = v_idx_all[2::4]
            v_idx[1::4] = v_idx_all[3::4]
            v_idx[2::4] = v_idx_all[0::4]
            v_idx[3::4] = v_idx_all[1::4]
            v= (v.flatten()[v_idx]).reshape(-1,4)

        if do_roll:
            shift_rand = torch.rand(1)
            shift = int((w * shift_rand ).tolist()[0])
            image = torch.roll(image , shift , 2 )            
            u = (u + shift_rand) % 1
        #============     Padding Data     ===========        
        u_grad = get_grad_u(u.flatten()[::2].reshape((-1,1)) , _width=self.padding_count , c = self.c)        
        u_grad = torch.max(u_grad,0)[0]  

        padding_count = (self.padding_count  - u.numel()//2)
        padding_count = max(padding_count ,  0)        
        
        padding_count = abs( self.padding_count *2 - u.numel())
        u_pad = torch.cat(( u.reshape(-1) , torch.zeros((padding_count )) )  )                
        du_pad = torch.cat(( du.reshape(-1) , torch.zeros((self.padding_count  - du.numel() )) )  )                        
        v= v.flatten()                
        padding_count = abs( self.padding_count  *4 - v.numel() )
        v_top_pad = torch.cat(( v[::2] , torch.zeros((padding_count//2 )) )  )        
        v_btm_pad = torch.cat(( v[1::2] , torch.zeros((padding_count//2 )) )  )      
        
        #====================================
        #             Wrap Data
        #====================================
        data['image'] = image        
        data['u_grad'] = u_grad
        data['u'] = u_pad [::2]
        data['v_top'] = v_top_pad[::2]
        data['v_btm'] = v_btm_pad[::2]

        data['du'] = du_pad
        
        data['dv_top'] = v_top_pad[1::2]
        data['dv_btm'] = v_btm_pad[1::2]
        
        #=====================
        #|    output shape   |
        #=====================
        #   u: [n, 2]
        #   u_grad: [n, width]   , width default = 1024
        #   v_top: [n]
        #   v_btm: [n]

        return data

# [ Test ]

if __name__=="__main__"    :
    
    dataset = CustomDataset( f"../anno/test_visiable_10_no_cross.json"  , use_aug= False )     
    dataloader = DataLoader(dataset, 2 , shuffle=False, drop_last =True)
    
    data = next(iter(dataloader))     
    vis = visualize_2d(     
        data['u'],
        data['v_top'],
        data['v_btm'],
        data['image'],
        data['u_grad'],
    )
    plt.imshow(vis[0])
    plt.show()
'''
    
    pack_gt = (data['u'] , data['v_top'] , data['v_btm'] , data['du'] , data['dv_top'] , data['dv_btm'] , data['u_grad']  )        
    pack_gt = torch.cat(pack_gt , 1)
    b, _ = pack_gt.shape
    pack_gt = pack_gt.reshape((b,7,-1))    
    pack_gt = encode(pack_gt)    
    
    u= pack_gt[0]
    v_top =pack_gt[1]
    v_btm = pack_gt[2]
    du = pack_gt[3]
    dv_top = pack_gt[4]
    dv_btm = pack_gt[5]
    
    u,vt,vb,scores =  decode((u , v_top , v_btm , du ,dv_top ,dv_btm),0.5 )
    visualize_2d(     
        u,
        vt,
        vb,
        data['image'],
        data['u_grad'],
    )
'''


