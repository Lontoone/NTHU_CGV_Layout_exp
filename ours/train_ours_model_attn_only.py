# Data Loader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import json
from CustomDataset import * 
import sys
sys.path.append('../')
from config import *
from file_helper import *
#from Horizon_and_SAM.Horizon import PE_helper
from  PE_helper import *

from pytorch_lightning.callbacks import ModelCheckpoint , Callback

def collate_fn(batch):
    return tuple(zip(*batch))
#=================================
#             Augmentation
#=================================

def gauss_noise_tensor(img):
    rand = torch.rand(1)[0]
    if rand < 0.5 and Horizon_AUG:
        sigma = rand *0.125
        out = img + sigma * torch.randn_like(img)
        return out
    return img

def blank(img):    
    return img

class CustomDataModule(pl.LightningDataModule):
    def __init__(self ,
                 train_dir ,
                 test_dir , batch_size = 2,
                 num_workers = 0 , img_size=[IMG_WIDTH, IMG_HEIGHT] , use_aug = True ,padding_count = 24 ,c =0.1
                   ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size      
        self.use_aug = use_aug
        self.padding_count  = padding_count
        self.c = c
        

        pass

    def prepare_data(self) -> None:
        # Download dataset
        pass

    def setup(self, stage):
        # Create dataset...          
                
        self.entire_dataset = CustomDataset(self.train_dir  , use_aug= self.use_aug , padding_count= self.padding_count , c=self.c , img_size=self.img_size)
        self.train_ds , self.val_ds = random_split(self.entire_dataset , [0.9, 0.1])        
        self.test_ds = CustomDataset(self.test_dir  , use_aug= False , img_size=self.img_size ,  padding_count= self.padding_count )
        
        print("image size ",self.img_size)
        pass

    # ToDo: Reture Dataloader...
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds , batch_size= self.batch_size , num_workers= self.num_workers , shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_ds , batch_size= self.batch_size , num_workers= self.num_workers , shuffle=False)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds , batch_size= self.batch_size , num_workers= self.num_workers , shuffle=True)

    pass



from torch import Tensor
def unpad_data( x :[Tensor] ) :
    non_zero_indices = torch.nonzero(x)
    #print(non_zero_indices)
    # Get the non-zero values
    non_zero_values = x[non_zero_indices[:,0], non_zero_indices[:,1]]

    unique = torch.unique(non_zero_indices[:,0] ,return_counts=True)    
    non_zero_values = torch.split(non_zero_values , tuple(unique[1]))
    
    return non_zero_values



import torch
from torch import nn
from torch.nn import functional as F
from typing import Any
import pytorch_lightning as pl
from config import *
import torchvision.models as models
from torchvision.ops import MLP
import math
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from VerticalCompressionNet import * 
from CustomTransformer import *

def encode_target(box_b , base_u):
    '''
    box_b[:, 1] = (0.5 -box_b[:, 1])  # v top
    box_b[:, 2] = (box_b[:, 2] -0.5) # v btm
    #box_b[:, 3] = ( box_b[:, 3] + base_u) # du
    box_b[:, 3] = box_b[:, 3] # du

    box_b[:, 4] = (0.5 -box_b[:, 4])  # v top
    box_b[:, 5] = (box_b[:, 5] -0.5) # v btm
    box_b[:, 0] = (base_u - box_b[:, 0])  # u
    '''
    
    box_b[:, 1] = torch.log(torch.abs(0.5 -box_b[:, 1]))  # v top
    box_b[:, 2] = torch.log(torch.abs(box_b[:, 2] -0.5)) # v btm
    box_b[:, 3] = torch.log(torch.abs(box_b[:, 3])) # du

    box_b[:, 4] = torch.log(torch.abs(0.5 -box_b[:, 4]))  # v top
    box_b[:, 5] = torch.log(torch.abs(box_b[:, 5] -0.5)) # v btm
    box_b[:, 0] = torch.log( base_u - box_b[:, 0]  )  # u

    return box_b
def decode_target(box_b , base_u):    
    '''
    box_b[:, 0] = base_u - box_b[:, 0]  # u
    box_b[:, 1] = 0.5 - box_b[:, 1]  # v top
    box_b[:, 2] = box_b[:, 2] +0.5 # v btm
    box_b[:, 3] = base_u + box_b[:, 3]  # du

    box_b[:, 4] = 0.5 -box_b[:, 4]  # v top    
    box_b[:, 5] = box_b[:, 5] +0.5 # v btm
    '''
    
    box_b[:, 0] = base_u - torch.exp( box_b[:, 0] )  # u
    box_b[:, 1] = 0.5 - torch.exp(box_b[:, 1])  # v top
    box_b[:, 2] = torch.exp(box_b[:, 2]) +0.5 # v btm
    box_b[:, 3] = torch.exp(box_b[:, 3]) + base_u  # du

    box_b[:, 4] = 0.5 - torch.exp(box_b[:, 4])  # v top    
    box_b[:, 5] = torch.exp(box_b[:, 5]) +0.5 # v btm
    return box_b

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: 256, dropout: float = 0.1, max_len: int = 1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model: 256, nhead: int , d_hid: int, nlayers: int, dropout: float = 0.1 , activation="relu" ,
                  normalize_before=False , out_dim=20 ,channel = 1024):
        super().__init__()
        #self.ntoken = ntoken
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.out_dim =out_dim
       
        self.enc_proj = nn.Linear(channel , self.out_dim * 4)
        
        self.cls_head = nn.Linear(channel, 1 )
        self.u_head = nn.Linear(channel, 2 )
        self.v_head = nn.Linear(channel, 4 )

        self.u_head.bias.data.fill_(self.out_dim /100*0.5)
        self.v_head.bias.data.fill_(0.15)

        
    
        
    def forward(self, src: Tensor ) -> Tensor:
        # permute to (Sequence_length , Batches , Hidden layer)
        '''
        plt.imshow(src[0].detach().cpu().numpy())
        plt.title("src")
        plt.show()
        '''
        src         = src.permute(2,0,1)   #  [w, b, c*h]  , example: [256 , 5 , 1024]        
        #src         = src.permute(1 , 0 , 2)# torch.Size([1024, b, 256])        
        batch_size  = src.shape[1]   
        '''
        src_pe         = self.pe(src)   # [ 256 , b , hidden_dim]
        #print("src_pe",src_pe.shape)
        src_pe         = self.encoder(src_pe) # [ 256 , b , hidden_dim]
        '''     
        src_pe         = self.enc_proj(src) # [ 256 , b , max count * 4
        src_pe         = src_pe.view(src_pe.shape[0], src_pe.shape[1], self.out_dim , 4) # [ 256 , b , max count * 4
        #src_pe         = src_pe.permute(1,0,2)  # [b , width , 1024]
        out = src_pe.permute(1 , 2 , 0, 3)  # [b, max_count , seq_len, step_cols]
        out = out.contiguous().view(out.shape[0] , self.out_dim , -1) 

        #print(src_pe.shape)
        #plt.imshow(src_pe[0].detach().cpu().numpy())
        #plt.title("encoder output")
        #plt.show()
        #print("self.query_embed.weight " , self.query_embed.weight .shape)
        #out = self.decoder( self.query_embed.weight , src_pe)
        '''
        out = self.decoder1(src_pe)        
        out = torch.relu(out)
        out = self.decoder2(out)        
        out = torch.relu(out)
        out = self.decoder3(out)        
        out = torch.relu(out)
        '''
        #print("out",src_pe.shape)        
       
        box_u_logits = self.u_head(out)
        box_v_logits = self.v_head(out)
        cls_logits = self.cls_head(out)
        
        #print("box_v_logits" , box_v_logits.shape)
        box_logits = torch.cat([ box_u_logits[:,:,0].unsqueeze(2) ,
                                 box_v_logits[:,:,0].unsqueeze(2) ,
                                 box_v_logits[:,:,1].unsqueeze(2) , 
                                 box_u_logits[:,:,1].unsqueeze(2) ,
                                 box_v_logits[:,:,2].unsqueeze(2) ,
                                 box_v_logits[:,:,3].unsqueeze(2)] , dim=-1 )
        #print("box_logits" , box_logits.shape)
        return box_logits ,cls_logits



    

class VerticalQueryTransformer(pl.LightningModule):    
    def __init__(self  , 
                    hidden_out = 128 , class_num = 1 ,
                    log_folder = "__test" , num_classes = 1 , 
                    backbone_trainable =False, load_weight =""  ,
                    dropout = 0.01 , normalize_before=False
                    ,stride = 3,
                    img_size = [1024,512]
                    ):
        #print(" input_size" ,  input_size)
        super().__init__()
        self.confidence_threshold = 0.8
        self.log_folder = create_folder(os.path.join(os.getcwd() , "output" , log_folder))
        self.automatic_optimization = False
        self.hidden_size = hidden_out
        self.num_classes  = num_classes 

        self.input_width = img_size[0]
        self.input_height = img_size[1]
        self.stride=stride
        self.patch_out =  self.input_width//self.stride
        
        

        #self.pixel_value_proj = nn.Linear( 3*self.input_height , self.hidden_size )

        self.pe = PositionalEncoding(self.hidden_size ,dropout , max_len=  self.input_width)
        '''
        encoder_norm = nn.LayerNorm(self.hidden_size) if normalize_before else None
        encoder_layer = TransformerEncoderLayer(self.hidden_size, 8, 2048,
                                                dropout, 'relu', normalize_before)
        self.encoder = TransformerEncoder(encoder_layer, 8, encoder_norm )
        '''
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, dropout=dropout, activation='relu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        #decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=8  )
        #self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)


        #self.backbone = Resnet()

        self.pos_emb = nn.Parameter(torch.randn(size=(1,  self.patch_out, hidden_out)), requires_grad=True)
        
        self.pixel_value_proj =nn.Conv2d(
                in_channels= 3 ,
                out_channels=hidden_out,
                kernel_size= (self.input_height , stride),
                stride= (self.input_height , stride),
            )
        self.mlp = MLP(hidden_out , [hidden_out//2 , hidden_out//4 ,hidden_out//8 , hidden_out//16 ])

        self.cls_head = nn.Linear(self.hidden_size//16 , 1 )
        self.u_head = nn.Linear(self.hidden_size//16, 2 )
        self.v_head = nn.Linear(self.hidden_size//16, 4 )

        self.u_head.bias.data.fill_(1 /self.patch_out*0.5)
        self.v_head.bias.data.fill_(0.25)
    def reset_eval_helper(self ):        
        self.eval_helper = PR_Eval_Helper()
        pass
    def forward(self ,x ):
        #x = self.backbone(x)[-1] 
        #x = x.permute(0,3,1,2)  # [ batch , width , channel , height]
        #x = x.view(x.shape[0] , x.shape[1] , -1)  # [ batch , width , channel * height]        
        
        pixel_feat = self.pixel_value_proj(x)  # [ batch , hidden , height(=1) , patches ]
        pixel_feat = pixel_feat.flatten(2).permute(0,2,1)   # [ batch , patches ,hidden ]
        
        src_pe =  pixel_feat + self.pos_emb  # [ batch , patches ,hidden ]
        src_pe =  self.pe(src_pe)
        
        enc_out = self.encoder(src_pe)     
        #enc_out , weight = self.encoder(src_pe , return_weight= True )             
        
        enc_out = self.mlp(enc_out)
        box_u_logits = self.u_head(enc_out)
        box_v_logits = self.v_head(enc_out)
        cls_logits = self.cls_head(enc_out)
        
        box_logits = torch.cat([ box_u_logits[:,:,0].unsqueeze(2) ,
                                 box_v_logits[:,:,0].unsqueeze(2) ,
                                 box_v_logits[:,:,1].unsqueeze(2) , 
                                 box_u_logits[:,:,1].unsqueeze(2) ,
                                 box_v_logits[:,:,2].unsqueeze(2) ,
                                 box_v_logits[:,:,3].unsqueeze(2)] , dim=-1 )
        
        return box_logits ,cls_logits            

        return out_box , out_cls
    
    @torch.no_grad()
    def __calculate_pixel_PR_curve(self , pred_b , gt_input ):

        gt_u_b = unpad_data( gt_input['u'])          
        gt_vtop_b =unpad_data(gt_input['v_top'])
        gt_vbtm_b = unpad_data (gt_input['v_btm'])
        gt_du_b = unpad_data(gt_input['du'])
        gt_dvtop_b = unpad_data(gt_input['dv_top'])
        gt_dv_btm_b = unpad_data(gt_input['dv_btm'])

        EVAL_PIXEL_MASK_WIDTH = 2048
        EVAL_PIXEL_MASK_HEIGHT = 1024

        #print("pred" , pred_b)
        #print("gt u" , gt_u_b)
        batch_size = len(gt_u_b)

        #print("batch_size" , batch_size)

        # Calculate IoU based on pixel level mask
                

        for _batch_cnt in range(batch_size):
            pred_masks = []
            gt_masks = []

            pred = pred_b[_batch_cnt] if len(pred_b) >0 else []
            for p in pred:
                
                p = p.detach().cpu().numpy()
                pred_u =  p[[0,3]].reshape(-1,2)
                pred_vt = p[[1,4]].reshape(-1,2)
                pred_vb = p[[2,5]].reshape(-1,2) 
                

                polys , pred_mask = to_distorted_box(pred_u,pred_vt,pred_vb , return_mask= True ,
                                                h =EVAL_PIXEL_MASK_HEIGHT ,
                                                w = EVAL_PIXEL_MASK_WIDTH , 
                                                seg_count=30 , show_plt=False)
                pred_masks.append(pred_mask)
                #plt.imshow(pred_mask)
                #plt.title("Pred")
                #plt.show()

            u,vtop,vbtm,du,dvtop, dvbtm  = (gt_u_b[_batch_cnt] , gt_vtop_b[_batch_cnt] , gt_vbtm_b[_batch_cnt] , gt_du_b[_batch_cnt] ,
                                             gt_dvtop_b[_batch_cnt] , gt_dv_btm_b[_batch_cnt]  )
            gt_box =  torch.vstack([ u, vtop,vbtm, u + du ,dvtop , dvbtm]).permute(1,0)   # [n , 6]        
            
            
            for box in gt_box:
            
                box = box.detach().cpu().numpy()
                gt_u =  box[[0,3]].reshape(-1,2)
                gt_vt = box[[1,4]].reshape(-1,2)
                gt_vb = box[[2,5]].reshape(-1,2) 
                polys , gt_mask = to_distorted_box(gt_u,gt_vt,gt_vb , return_mask= True ,
                                                h =EVAL_PIXEL_MASK_HEIGHT ,
                                                w = EVAL_PIXEL_MASK_WIDTH , 
                                                seg_count=30 , show_plt=False)
                gt_masks.append(gt_mask)
                #plt.imshow(gt_mask)
                #plt.title("gt_mask")
                #plt.show()

            self.eval_helper.eval_batch_pr(pred_masks ,gt_masks  )



        pass
    
    @torch.no_grad()
    def __calculate_pr(self):
        assert self.eval_helper is not None , "eval_helper not reset"
        self.eval_helper.get_all_pr(self)

        self.eval_helper = None
        pass
    @torch.no_grad()
    def inf(self , imgs ):
        decoded_pred_outputs =[]
        out_box , out_cls   = self.forward(imgs)  # [ batch , top_k , 5]   , [ batch , top_k , 1]                         
        
        batch_size = out_box.shape[0]
        
        #each batch
        for img , pred , pcls in zip(imgs, out_box , out_cls.view(batch_size,-1)):
            
            u_id = torch.argwhere(torch.sigmoid(pcls) > self.confidence_threshold)
            if(u_id.numel() ==0):
                continue
            u_id = u_id.view(-1)            
            
            #pred = self.post_process(pbox[u_id,:] , u_id ).view(-1,6)
            pred = pred[u_id]
            u_grad = (torch.arange(self.patch_out,device=pred.device).unsqueeze(-1) /self.patch_out ).view(-1)[u_id]
            
            save_folder = create_folder( os.path.join(self.log_folder ,"val"))
            save_path = os.path.join(save_folder, f"val_ep_{self.current_epoch}-{self.global_step}" )

            decode_pred = decode_target(pred.clone() , u_grad )
            decoded_pred_outputs.append(decode_pred)
            
            pred_us , pred_tops , pred_btms = self.pack_visualize(decode_pred[:,0], decode_pred[:,1],decode_pred[:,2],decode_pred[:,3]-decode_pred[:,0] ,decode_pred[:,4],decode_pred[:,5] )                    
            vis_imgs = visualize_2d_single(pred_us , pred_tops , pred_btms , u_grad = F.sigmoid(pcls).view(1 , -1 ) , imgs=  img , title="Pred" , save_path= save_path  )
            
            #plt.imshow(vis_imgs)
            #plt.title("inference")
            #plt.show()

        return out_box , out_cls , decoded_pred_outputs


        
    @torch.no_grad()
    def pack_visualize(self, gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , dv_btm_b ):
        
        if isinstance(gt_u_b, torch.Tensor):
            sizes = [t.numel() for t in gt_u_b]               
            us = gt_u_b.flatten().unsqueeze(0).repeat(2, 1).permute(1,0).reshape(-1)
            us[1::2]+=gt_du_b.flatten()
            us = torch.split(us.view(-1,2) , sizes)

            tops = gt_vtop_b.flatten().unsqueeze(0).repeat(2, 1).permute(1,0).reshape(-1)
            tops[1::2]=gt_dvtop_b.flatten()
            tops = torch.split(tops.view(-1,2) , sizes)

            btms = gt_vbtm_b.flatten().unsqueeze(0).repeat(2, 1).permute(1,0).reshape(-1)
            btms[1::2]=dv_btm_b.flatten()
            btms = torch.split(btms.view(-1,2) , sizes)

        elif isinstance(gt_u_b, tuple) and all(isinstance(t, torch.Tensor) for t in gt_u_b):        
            sizes = [len(t) for t in gt_u_b]               
            us = torch.cat(gt_u_b).view(-1).unsqueeze(0).repeat(2, 1).permute(1,0).reshape(-1)
            us[1::2]+=torch.cat(gt_du_b).view(-1)
            us = torch.split(us.view(-1,2) , sizes)

            tops = torch.cat(gt_vtop_b).view(-1).unsqueeze(0).repeat(2, 1).permute(1,0).reshape(-1)
            tops[1::2]=torch.cat(gt_dvtop_b).view(-1)
            tops = torch.split(tops.view(-1,2) , sizes)

            btms = torch.cat(gt_vbtm_b).view(-1).unsqueeze(0).repeat(2, 1).permute(1,0).reshape(-1)
            btms[1::2]=torch.cat(dv_btm_b).view(-1)
            btms = torch.split(btms.view(-1,2) , sizes)
        else:
            assert("Wrong Type.")
        
        return us , tops ,btms
        
    def __common_step( self ,batch_idx,input_b , out_box , out_cls , mode ="train"):        
        
        assert mode in ['train' , 'val' , 'test'] , "Mode have to be [train , val , test]"
        # remove padding , each batch have different length
        img = input_b['image']
        gt_u_b = unpad_data( input_b['u'])          
        gt_vtop_b =unpad_data(input_b['v_top'])
        gt_vbtm_b = unpad_data (input_b['v_btm'])
        gt_du_b = unpad_data(input_b['du'])
        gt_dvtop_b = unpad_data(input_b['dv_top'])
        gt_dv_btm_b = unpad_data(input_b['dv_btm'])
        
        total_loss = 0
        b_cnt = 0        
        
        for u,vtop,vbtm,du,dvtop, dvbtm , pred ,cls_b , gt_grad_cls  in zip(gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , gt_dv_btm_b , out_box , out_cls ,input_b['u_grad'] ):
            
            # match                        
            gt_box =  torch.vstack([ u, vtop,vbtm,  du ,dvtop , dvbtm]).permute(1,0)   # [n , 6]            
            # gt u to grad
            u_grad = (torch.arange(self.patch_out,device=u.device).unsqueeze(-1) /self.patch_out ).view(-1)            
            # decode prediction (turn u and v to offsets)
            decode_pred_b = decode_target(pred.clone() , u_grad)           
            
            # When training , use gt to guide. 
            # ---------- Cost Matrix ----------
            u_cost = torch.cdist( (torch.arange(self.patch_out,device=u.device).unsqueeze(-1) /self.patch_out ) , u.unsqueeze(-1) )                                    
            cls_cost = -torch.sigmoid(cls_b)            
            
            cost_matrix =  cls_cost + u_cost * 2            
            

            # Select by cost matrix
            cost_matrix = cost_matrix.detach().cpu().numpy()            
            row_idx  , col_idx = linear_sum_assignment(cost_matrix)    

            matched_u = torch.tensor( np.float32(row_idx)/self.patch_out,device=u.device)
            # encode gt box based on selected u
            encode_gt_b = encode_target(gt_box.clone()  , matched_u)
                        
            #gt_cls = torch.zeros(self.patch_out,device= cls_b.device )            
            #gt_cls[row_idx] = 1            
            
            # Loss 
            l1_loss = F.l1_loss(pred[row_idx] ,  encode_gt_b[col_idx])             
            cls_loss = F.binary_cross_entropy_with_logits(cls_b.view(-1), gt_grad_cls) 
                        
            total_loss += l1_loss + cls_loss*5            

            self.log(f"{mode}/l1_loss" , l1_loss)
            self.log(f"{mode}/cls_loss" , cls_loss)
            self.log(f"{mode}/total_loss" , total_loss)
                        
            with torch.no_grad():
                #if self.current_epoch % 5 == 0  :                
                #if self.current_epoch % 5 == 0 and self.current_epoch > 0 and batch_idx<2 :                
                if self.current_epoch > 0 and self.current_epoch % 5 == 0  and batch_idx <5 :                
                    save_path =  os.path.join(self.log_folder , f"gt_ep_{self.current_epoch}-{self.global_step}-{batch_idx}" )
                    
                    gt_us , gt_tops , gt_btms = self.pack_visualize(u.view(1 , -1 ) , vtop , vbtm , du , dvtop , dvbtm )                   

                    vis_imgs = visualize_2d_single(gt_us , gt_tops , gt_btms , u_grad =  gt_grad_cls.view(1 , -1 ), imgs= img[b_cnt] , title="GT",save_path=save_path )                
                    
                    decode_pred = decode_pred_b[row_idx]
                    
                    save_path =  os.path.join(self.log_folder , f"pred_ep_{self.current_epoch}-{self.global_step}-{batch_idx}" )                    
                    pred_us , pred_tops , pred_btms = self.pack_visualize(decode_pred[:,0], decode_pred[:,1],decode_pred[:,2],
                                                                          decode_pred[:,3]-decode_pred[:,0] ,
                                                                          decode_pred[:,4],decode_pred[:,5] )                    
                    vis_imgs = visualize_2d_single(pred_us , pred_tops , pred_btms , u_grad = F.sigmoid(cls_b).view(1 , -1 ) , imgs=  img[b_cnt] ,
                                                    title=f"Pred_row{row_idx}-\n u:{pred_us}" , save_path= save_path  )
                    plt.imshow(vis_imgs)                    
                    plt.show()

            b_cnt+=1
        return total_loss

    def training_step(self , input_b ,batch_idx ):
        
        img = input_b['image']        
        out_box , out_cls   = self.forward(img)  # [ batch , top_k , 5]   , [ batch , top_k , 1] 
        batch_size = out_box.shape[0]
        total_loss = self.__common_step(batch_idx , input_b , out_box , out_cls , "train")        
        
        op1  = self.optimizers()
        op1.zero_grad()        
        self.manual_backward(total_loss / batch_size)
        op1.step()

        return total_loss / batch_size
        pass    

    def validation_step(self, input_b, batch_idx):
        #print("val!!!!!")
        img = input_b['image']
        
        #out_box , out_cls   = self.forward(img)  # [ batch , top_k , 5]   , [ batch , top_k , 1]         
        #if( self.current_epoch %5==0 and batch_idx % 10==0 and self.current_epoch>0 ):
            #self.inf(img)
        out_box , out_cls   = self.forward(img)  
        self. __common_step(batch_idx , input_b ,out_box, out_cls , "val" )
        return
        

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters() , lr=0.00035)

        return [opt] , []
    
    def on_test_epoch_start(self):
        self.reset_eval_helper()

    def test_step(self , input_b , batch_idx):
        
        img = input_b['image']        
        
        #out_box , out_cls   = self.forward(img)  # [ batch , top_k , 5]   , [ batch , top_k , 1]         
        #if( self.current_epoch %5==0 and batch_idx % 10==0 and self.current_epoch>0 ):
            #self.inf(img)
        out_box , out_cls ,decoded_pred_outputs  = self.inf(img)        
        self. __common_step(batch_idx , input_b ,out_box, out_cls , "test" )

        self.__calculate_pixel_PR_curve(decoded_pred_outputs , input_b)
    
    def on_test_epoch_end(self):
        self.__calculate_pr()

    pass

from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
if __name__ =="__main__":
    parser = ArgumentParser()    
    parser.add_argument("-gpu_num", default= 1, dest="gpu_num")
    parser.add_argument("-test", default= True, dest="test")
    parser.add_argument("-b", default= 6, dest="batch_size")  #batch size
    parser.add_argument("-w", default= 256, dest="img_w")    
    parser.add_argument("-h", default= 128, dest="img_h")    
    parser.add_argument("-s", default= 3, dest="vert_conv_strid")    
    args = parser.parse_args()
    

    img_size=[args.img_w , args.img_h] 
    padding = args.img_w // args.vert_conv_strid
    # Test
    dm = CustomDataModule ( train_dir= f"../anno/train_visiable_20_no_cross.json" ,
                            test_dir= f"../anno/test_visiable_10_no_cross.json" ,
                            #test_dir= f"../anno/train_visiable_20_no_cross.json" ,
                            padding_count=padding , use_aug=False , c= 0.65,batch_size=args.batch_size,
                            img_size=img_size
                        )
    
    logger = TensorBoardLogger('tb_logs', name='encoder_only_test')

    m = VerticalQueryTransformer(
                                 hidden_out=1024 ,
                                 img_size=img_size  ,
                                 backbone_trainable=True ,
                                 dropout=0.01 , 
                                 stride= args.vert_conv_strid
                                 )

    save_path = create_folder( os.path.join(os.getcwd() , "output" , "checkpoints"))
    '''
    save_path = create_folder( os.path.join(os.getcwd() , "output" , "checkpoints"))
    save_file = os.path.join(save_path , "detr_v1_d20_e50.pth")
    m = torch.load(save_file)
            


    save_file = os.path.join(save_path , "detr_v9_d20_e200_bugfix_pe_overfit.pth")
    #save_file = os.path.join(save_path , "detr_v9_d20_e200_bugfix_pe_customAtten.pth")
    m.load_state_dict(torch.load(save_file))
    '''


    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',  # The validation metric to monitor
        dirpath= save_path ,  # Directory where checkpoints will be saved
        filename='best-model-{epoch:02d}-{val_loss:.2f}',  # Checkpoint file name
        save_top_k=3,  # Save only the best model
        mode='min'  # 'min' for metrics where lower is better (like loss), 'max' for metrics where higher is better (like accuracy)
    )

    class MyCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is about to start!")

    trainer = pl.Trainer(accelerator='gpu' , devices=args.gpu_num ,
                        min_epochs=1, max_epochs=201 , precision=32 , logger= logger , fast_dev_run=args.test  ,
                        callbacks=[checkpoint_callback , MyCallback()])
    trainer.fit(m , dm)
    trainer.test(m , dm)


