# Data Loader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
#from torch.utils.data import random_split
import json
#from CustomDataset import * 
from Horizon_DataLoader import * 
import sys
sys.path.append('../')
from config import *
from file_helper import *
#from Horizon_and_SAM.Horizon import PE_helper
from  PE_helper import *

from pytorch_lightning.callbacks import ModelCheckpoint , Callback

def cornersToxyxy( coroner_u ,  corners_b ):
    # Input
    #   corners_b : shape [n , 1 ] , format u
    #   corners_b : shape [n , 5 ] , format vvuvv
    # Output
    #   shape [ n , 4 ] , format xyxy
    n = corners_b.shape[0]
    #results = torch.zeros((b, n , 4) ,device= corners_b.device)
    results = torch.zeros((n , 4) ,device= corners_b.device)
    results[:, 0] = coroner_u[ :, 0 ]
    results[:, 1] = torch.min(corners_b[:, 0 ] , corners_b[:, 3 ]  )
    results[:, 2] = coroner_u[ :, 0 ] + corners_b[:, 2 ]
    results[:, 3] = torch.min(corners_b[:, 1 ] , corners_b[:,4 ]  )

    return results



    
def collate_fn(batch):
    return tuple(zip(*batch))
#=================================
#             Augmentation
#=================================

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
                
        self.entire_dataset = CustomDataset(self.train_dir  , use_aug= self.use_aug ,  c=self.c , img_size=self.img_size)
        self.train_ds , self.val_ds = random_split(self.entire_dataset , [0.99, 0.01])        
        self.test_ds = CustomDataset(self.test_dir  , use_aug= False , img_size=self.img_size  )
        
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



from scipy.ndimage.filters import maximum_filter
def find_N_peaks2(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    if(min_v>0):
        pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]

from raw_model_pixelwize_bbox import *
from Horizon_DataLoader import  * 
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from torch import Tensor
def unpad_data( x :[Tensor] ) :
    non_zero_indices = torch.nonzero(x)
    #print(non_zero_indices)
    # Get the non-zero values
    non_zero_values = x[non_zero_indices[:,0], non_zero_indices[:,1]]

    unique = torch.unique(non_zero_indices[:,0] ,return_counts=True)    
    non_zero_values = torch.split(non_zero_values , tuple(unique[1]))
    
    return non_zero_values
@torch.no_grad()
def pack_visualize( gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , dv_btm_b ):
    
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

class Model(pl.LightningModule):    
    def __init__(self  , output_log_name = "out"):
        super().__init__()
        self.model = HorizonNet(backbone='resnet50', use_rnn=True)
        self.log_folder = create_folder( os.path.join(os.getcwd() , "naive_direct_0301" ))
        self.post_thrshold = 0.5
        self.nms_threshold = 0.25

        # for visualization
        self.output_log_name = output_log_name
        self.output_log =[]
        
    def forward(self , x):
        prob , boxs  = self.model(x)  # [ b , _ , 1024]
        prob = prob.permute((0,2,1))
        boxs = boxs.permute((0,2,1))
        

        return prob , boxs
        pass
    def __common_stepup(self , input_b):
        img = input_b['image']        

        gt_u_b = unpad_data( input_b['u'])          
        gt_vtop_b =unpad_data(input_b['v_top'])
        gt_vbtm_b = unpad_data (input_b['v_btm'])
        gt_du_b = unpad_data(input_b['du'])
        gt_dvtop_b = unpad_data(input_b['dv_top'])
        gt_dv_btm_b = unpad_data(input_b['dv_btm'])

        out_prob , out_boxs = self.forward(img)
        batch_size = out_prob.shape[0]

        return gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , gt_dv_btm_b , out_prob ,out_boxs, batch_size
        pass
    def training_step(self , input_b ,batch_idx ):
        img = input_b['image']
        
        gt_pro = input_b['u_grad']
        gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , gt_dv_btm_b , out_prob ,out_boxs, batch_size = self.__common_stepup(input_b)
        
        total_loss = 0
        b_cnt = 0
        for u,vtop,vbtm,du,dvtop, dvbtm , pred_cls , pred_box , gt_cls in zip(gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , gt_dv_btm_b , out_prob , out_boxs , gt_pro):
            gt_box =  torch.vstack([ u, vtop,vbtm,  du ,dvtop , dvbtm]).permute(1,0)   # [n , 6]  
            gt_cnt = gt_box.shape[0]

            # Select n peak column
            #pkloc , signal = find_N_peaks2( pred_cls.detach().cpu().numpy().astype(np.float32).flatten() , min_v=-2, N=gt_cnt)            
            #pred_u = pkloc / 1024
            row_idx = (u * 1024).to(torch.long)  # gt_u_idx
            col_idx = torch.arange(gt_cnt).to(torch.long) 
            pred_u = row_idx / 1024
            
            # Match gt for each selected peak
            #u_cdist =distance.cdist( np.expand_dims(pred_u, axis=-1),  u.unsqueeze(-1).detach().cpu().numpy() )
            #row_idx , col_idx = linear_sum_assignment ( u_cdist )

            #l1_loss = F.l1_loss(pred_box[pkloc][row_idx] , gt_box[col_idx,1:]   )         
            l1_loss = F.l1_loss(pred_box[row_idx] , gt_box[col_idx,1:]   )         
            total_loss += l1_loss / gt_cnt

            with torch.no_grad():
                #if batch_idx % 5 ==0:
                if self.current_epoch > 0 and self.current_epoch % 5 == 0  and batch_idx <5 :                
                    save_path =  os.path.join(self.log_folder , f"gt_ep_{self.current_epoch}-{self.global_step}-{batch_idx}" )                    
                    # View GT
                    gt_us , gt_tops , gt_btms = pack_visualize(u.view(1 , -1 ) , vtop , vbtm , du , dvtop , dvbtm )                   
                    vis_imgs = visualize_2d_single(gt_us , gt_tops , gt_btms , u_grad =  gt_cls.view(1 , -1 ), imgs= img[b_cnt] , title="GT",save_path=save_path )                
                    #plt.imshow(vis_imgs)                    
                    #plt.show()
                    
                    # View Prediction
                    #decode_pred = decode_pred_b[row_idx]
                    #decode_pred = pred_box[pkloc][row_idx].detach().cpu()
                    decode_pred = pred_box[row_idx]#.detach().cpu()
                    save_path =  os.path.join(self.log_folder , f"pred_ep_{self.current_epoch}-{self.global_step}-{batch_idx}" )
                    pred_us , pred_tops , pred_btms = pack_visualize( torch.as_tensor(pred_u) , decode_pred[:,0],decode_pred[:,1],
                                                                          decode_pred[:,2] ,decode_pred[:,3],decode_pred[:,4] )                    
                    vis_imgs = visualize_2d_single(pred_us , pred_tops , pred_btms , u_grad = F.sigmoid(pred_cls).view(1 , -1 ) , imgs=  img[b_cnt] ,
                                                    title=f"Pred_row{row_idx}-\n u:{pred_us}" , save_path= save_path  )
                    #plt.imshow(vis_imgs)                    
                    #plt.show()
                pass
            b_cnt+=1
        total_loss/= batch_size
        
        cls_loss = F.binary_cross_entropy_with_logits(out_prob.view(batch_size , -1) , gt_pro.view(batch_size , -1))
        self.log(f"train_cls_loss" , cls_loss)
        self.log(f"train_box_loss" , total_loss)
        total_loss += cls_loss 
        self.log(f"train_total_loss" , total_loss)
        #print("cls_loss" , cls_loss)
        #print("total_loss" , total_loss)
        return total_loss
    
    def validation_step(self, input_b, batch_idx):        
        img = input_b['image']
        
        gt_pro = input_b['u_grad']
        gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , gt_dv_btm_b , out_prob ,out_boxs, batch_size = self.__common_stepup(input_b)
        
        total_loss = 0
        b_cnt = 0
        for u,vtop,vbtm,du,dvtop, dvbtm , pred_cls , pred_box , gt_cls in zip(gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , gt_dv_btm_b , out_prob , out_boxs , gt_pro):
            gt_box =  torch.vstack([ u, vtop,vbtm,  du ,dvtop , dvbtm]).permute(1,0)   # [n , 6]  
            gt_cnt = gt_box.shape[0]

            # Select n peak column
            #pkloc , signal = find_N_peaks2( pred_cls.detach().cpu().numpy().astype(np.float32).flatten() , min_v=self.post_thrshold, N=None)            
            #pred_u = pkloc / 1024
            row_idx = (u * 1024).to(torch.long)  # gt_u_idx
            col_idx = torch.arange(gt_cnt).to(torch.long) 
            pred_u = row_idx / 1024
            
            # Match gt for each selected peak
            #u_cdist =distance.cdist( np.expand_dims(pred_u, axis=-1),  u.unsqueeze(-1).detach().cpu().numpy() )
            #row_idx , col_idx = linear_sum_assignment ( u_cdist )

            l1_loss = F.l1_loss(pred_box[row_idx] , gt_box[col_idx,1:]   )         
            total_loss += l1_loss / row_idx.shape[0] +  abs(gt_cnt - row_idx.shape[0])            
            with torch.no_grad():
                #if batch_idx % 5 ==0:
                if self.current_epoch > 0 and self.current_epoch % 5 == 0  and batch_idx <5 :                
                    save_path =  os.path.join(self.log_folder , f"val_gt_ep_{self.current_epoch}-{self.global_step}-{batch_idx}" )                    
                    # View GT
                    gt_us , gt_tops , gt_btms = pack_visualize(u.view(1 , -1 ) , vtop , vbtm , du , dvtop , dvbtm )                   
                    vis_imgs = visualize_2d_single(gt_us , gt_tops , gt_btms , u_grad =  gt_cls.view(1 , -1 ), imgs= img[b_cnt] , title="GT",save_path=save_path )                
                    #plt.imshow(vis_imgs)                    
                    #plt.show()
                    
                    # View Prediction
                    #decode_pred = decode_pred_b[row_idx]
                    decode_pred = pred_box[row_idx]
                    save_path =  os.path.join(self.log_folder , f"val_pred_ep_{self.current_epoch}-{self.global_step}-{batch_idx}" )
                    pred_us , pred_tops , pred_btms = pack_visualize( torch.as_tensor(pred_u) , decode_pred[:,0],decode_pred[:,1],
                                                                          decode_pred[:,2] ,decode_pred[:,3],decode_pred[:,4] )                    
                    vis_imgs = visualize_2d_single(pred_us , pred_tops , pred_btms , u_grad = F.sigmoid(pred_cls).view(1 , -1 ) , imgs=  img[b_cnt] ,
                                                    title=f"Pred_row{row_idx}-\n u:{pred_us}" , save_path= save_path  )
                    #plt.imshow(vis_imgs)                    
                    #plt.show()
                pass
            b_cnt+=1
        total_loss/= batch_size
        
        cls_loss = F.binary_cross_entropy_with_logits(out_prob.view(batch_size , -1) , gt_pro.view(batch_size , -1))
        self.log(f"val_cls_loss" , cls_loss)
        self.log(f"val_box_loss" , total_loss)
        total_loss += cls_loss 
        self.log(f"val_total_loss" , total_loss)
        
        print("[val] total_loss" , total_loss)
        return total_loss
    
    def reset_eval_helper(self ):        
        self.eval_helper = PR_Eval_Helper(get_iou_fn= get_iou_matrix_polygon)
        pass
    @torch.no_grad()
    def __calculate_pr(self):
        assert self.eval_helper is not None , "eval_helper not reset"
        self.eval_helper.get_all_pr(self)

        self.eval_helper = None
        pass

    def on_test_epoch_start(self):
        self.output_log = []
        self.reset_eval_helper()

    def on_test_epoch_end(self):
        self.__calculate_pr()

        # write output log file
        log_save_folder = create_folder(os.path.join(os.getcwd() , "output_log" ))
        #log_save_file = os.path.join(log_save_folder , f"{self.output_log_name}.json")
        log_save_file = os.path.join(log_save_folder , f"naive_all_predict_out.json")
        
        json_object = json.dumps(self.output_log)         
        with open(log_save_file, "w") as outfile:
            outfile.write(json_object)


    @torch.no_grad()
    def test_step(self, input_b, batch_idx):
        EVAL_PIXEL_MASK_WIDTH = 1024
        EVAL_PIXEL_MASK_HEIGHT = 512

        imgs = input_b['image']        
        gt_pro = input_b['u_grad']
        img_paths =  input_b['image_path']
        
        gt_u_b , gt_vtop_b , gt_vbtm_b , gt_du_b , gt_dvtop_b , gt_dv_btm_b , out_prob ,out_boxs, batch_size = self.__common_stepup(input_b)
        b_cnt = 0

        # [ Prediction Output ]        
        for b_cnt , img in enumerate( imgs):
            predictions= []
            gts= []
        
            pred_box = out_boxs[b_cnt]
            pred_cls = out_prob[b_cnt]
            # Select n peak column
            #pkloc , signal = find_N_peaks2( pred_cls.detach().cpu().numpy().astype(np.float32).flatten() , min_v=self.post_thrshold, N=None)                        
            #pred_u = pkloc / 1024
            
            pkloc = torch.where(pred_cls > self.post_thrshold)[0]
            pred_u = (pkloc / 1024)
            #print("pkloc" ,pkloc)
            
            # View Prediction---------------------------------------------------------------
            ''''''
            decode_pred = pred_box[pkloc]     
            # [NMS]
            xyxy_box = cornersToxyxy(pred_u.reshape(-1 , 1) , decode_pred)  # [n , 4]
            nms_idx = torchvision.ops.nms(xyxy_box , pred_cls[pkloc].flatten() , self.nms_threshold)
            decode_pred = decode_pred[nms_idx]  .detach().cpu()     
            pred_u = pred_u[nms_idx]
            
            save_path =  os.path.join(self.log_folder , f"val_pred_ep_{self.current_epoch}-{self.global_step}-{batch_idx}" )
            pred_us , pred_tops , pred_btms = pack_visualize( torch.as_tensor(pred_u[:]).cpu() , decode_pred[:,0],decode_pred[:,1],
                                                                          decode_pred[:,2] ,decode_pred[:,3],decode_pred[:,4] )                    
            '''
            vis_imgs = visualize_2d_single(pred_us , pred_tops , pred_btms , u_grad = F.sigmoid(pred_cls).view(1 , -1 ) , imgs=  img ,
                                                    title=f"Pred_row{pkloc}-\n u:{pred_us}" , save_path= save_path  )
            plt.imshow(vis_imgs)                 
            plt.title("pred vis")   
            plt.show()

            '''
            debug_img = np.ascontiguousarray( img.permute(1,2,0).detach().cpu().numpy())

            for u , vt,vb in zip( pred_us , pred_tops , pred_btms):                                            
                #polys , debug_img  = to_distorted_box(u,vt,vb , return_mask= False ,
                polys  = to_distorted_box( u,vt,vb , return_mask= False ,
                                                h =EVAL_PIXEL_MASK_HEIGHT ,
                                                w = EVAL_PIXEL_MASK_WIDTH , 
                                                seg_count=30 , show_plt=False,
                                                image= debug_img,
                                                #seg_count=30 , show_plt=True , return_img=True , fill_color=(255,0,0)
                                        )
                predictions.append(polys)
            


            # [GT] ---------------------------------------------------------------
            u,vtop,vbtm,du,dvtop, dvbtm  = (gt_u_b[b_cnt] , gt_vtop_b[b_cnt] , gt_vbtm_b[b_cnt] , gt_du_b[b_cnt] , gt_dvtop_b[b_cnt] , gt_dv_btm_b[b_cnt]  )
            gt_box =  torch.vstack([ u, vtop,vbtm, u + du ,dvtop , dvbtm]).permute(1,0)   # [n , 6]                  
            
            gt_us , gt_tops , gt_btms = pack_visualize(u.view(1 , -1 ) , vtop , vbtm , du , dvtop , dvbtm )                   
            
            '''
            vis_imgs = visualize_2d_single(gt_us , gt_tops , gt_btms , u_grad =  gt_pro[b_cnt].view(1 , -1 ), imgs= img , title="GT",save_path=save_path )                
            plt.imshow(vis_imgs)                
            plt.title("gt vis")    
            plt.show()
            '''

            for box in gt_box:            
                box = box.detach().cpu().numpy()
                gt_u =  box[[0,3]].reshape(-1,2)
                gt_vt = box[[1,4]].reshape(-1,2)
                gt_vb = box[[2,5]].reshape(-1,2) 
                polys  = to_distorted_box(gt_u,gt_vt,gt_vb , return_mask= False ,
                #polys ,debug_img  = to_distorted_box(gt_u,gt_vt,gt_vb , return_mask= False ,
                                                h =EVAL_PIXEL_MASK_HEIGHT ,
                                                w = EVAL_PIXEL_MASK_WIDTH , 
                                                image= debug_img,
                                                seg_count=30 , show_plt=False
                                                #seg_count=30 , show_plt=True , return_img=True , fill_color=(0,255,0)
                                                
                                                )            
                gts.append(polys)

            self.eval_helper.eval_batch_pr(predictions , gts  )

            output_log={
                "img":img_paths[b_cnt],
                "gt_us":[aa.detach().cpu().numpy().tolist() for aa in gt_us],
                "gt_tops":[aa.detach().cpu().numpy().tolist() for aa in gt_tops],
                "gt_btms":[aa.detach().cpu().numpy().tolist() for aa in gt_btms],
                "pred_us":[aa.detach().cpu().numpy().tolist() for aa in pred_us],
                "pred_tops":[aa.detach().cpu().numpy().tolist() for aa in pred_tops],
                "pred_btms":[aa.detach().cpu().numpy().tolist() for aa in pred_btms]
            }
            #print("self.output_log" , len(self.output_log))
            self.output_log.append(output_log)
        return
    
    def configure_optimizers(self):
        #ToDo: Look at paper 
        opt = optim.Adam(self.parameters() , lr=0.00035 , betas= (0.9 , 0.999) , weight_decay= 0)
        # Ref: https://github.com/sunset1995/HorizonNet/blob/master/train.py#L44

        return [opt] , []
 

from pytorch_lightning.loggers import TensorBoardLogger
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model")
    #parser.add_argument('--logName', type=str,default='__test', help='Run logger name')
    #parser.add_argument('--load', type=str,default='', help='model path')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--ep', type=int, default=150)
    parser.add_argument('--b', type=int, default=6)   # batch size
    parser.add_argument('--t', action='store_true' )  # test
    #parser.add_argument('--r', action='store_true' )  # resume    
    #parser.add_argument('--itr', type=int, default=10)   # loop count
    #parser.add_argument('--citr', type=int, default=0)   # current loop count
    
    
    args = parser.parse_args()
    logger = TensorBoardLogger('tb_logs_0228', name='Horizon_naive_0301_all')

    dm = CustomDataModule ( train_dir= f"../../anno/train_visiable_all.json" ,
                            test_dir= f"../../anno/test_visiable_all.json" ,
                            #test_dir= f"../../anno/vis_from_test.json" ,
                            #test_dir= f"../anno/train_visiable_20_no_cross.json" ,
                            padding_count=1024,
                            use_aug=False , c= 0.96,batch_size= args.b * args.gpu,
                            img_size=[1024,512]
                        )


    #save_path = create_folder( os.path.join(os.getcwd() , "output" , "checkpoints"))
    save_path = create_folder( os.path.join(os.getcwd() , "naive_direct_0301" , "checkpoints"))
    #save_file = os.path.join(save_path , "naive_direct_ep14.ckpt")
    save_file = os.path.join(save_path , "naive_direct_best-model-epoch=68.ckpt")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_loss',  # The validation metric to monitor
        dirpath= save_path ,  # Directory where checkpoints will be saved
        filename='_debug_best-model-{epoch:02d}',  # Checkpoint file name
        save_top_k=3,  # Save only the best model
        mode='min',  # 'min' for metrics where lower is better (like loss), 'max' for metrics where higher is better (like accuracy)
        save_last=True
    )
    trainer = pl.Trainer(accelerator='gpu' , devices=args.gpu ,
                        min_epochs=1, max_epochs=args.ep , precision=16 , 
                        #strategy="ddp" ,
                        fast_dev_run=args.t, logger= logger,
                        #callbacks=[checkpoint_callback]
                        )
    m=Model()
    m = m.load_from_checkpoint(save_file)   # Load Model
    #trainer.fit(m , dm)
    trainer.test(m , dm)
    

# python OriginHorizon_0227.py --gpu 4
# python OriginHorizon_0227.py --gpu 1  --t
# python NaiveHorizon_0301_forVis.py --t