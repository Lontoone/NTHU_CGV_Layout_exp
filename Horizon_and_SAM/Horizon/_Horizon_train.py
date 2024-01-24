import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from sklearn import metrics
import numpy as np
import cv2
import sys
sys.path.append('../../')
from config import *
from PE_helper import *
from file_helper import *
from horizon_utlis import *
from Horizon_DataLoader import *
from horizon_model_direct import HorizonNet
from tqdm import tqdm

def create_model(load_pth = ""):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	net = HorizonNet('resnet50', True , MAX_PREDICTION_COUNT).to(device)   # For server (small memory)

	if (load_pth is not "" or None):              
		model_path = os.path.join(MODEL_FOLDER , load_pth)
		state_dict = torch.load(model_path, map_location='cpu')
		
		# model data
		net.load_state_dict(state_dict['state_dict'])
		LOADED_EPOCH    = state_dict['epoch']
		LOADED_AP       = state_dict['ap']
		save_auc        = LOADED_AP  
		print( LOADED_EPOCH , LOADED_AP)
	return net

def train_loop( model , opt ,dataloader , ep_count = 0  , log_folder ="", device = 'cuda'):
    
    it_count = 0    
    model.train()
    for data in tqdm(dataloader , desc="ep "+str(ep_count) , ):       

        for k, v in data.items():    
            data[k]=data[k].to(device)       
            
        pack_gt = (data['u'] , data['v_top'] , data['v_btm'] , data['du'] , data['dv_top'] , data['dv_btm'] , data['u_grad']  )        
        pack_gt = torch.cat(pack_gt , 1)
        b, _ = pack_gt.shape
        pack_gt = pack_gt.reshape((b,7,-1))
        #pack_gt = pack_gt.reshape((b,6,-1))
        pack_gt = encode(pack_gt)        
        
        #out = predict(data , model)   #[b , max_count , 5 ]
        out = model(data['image'])   #[b , max_count , 5 ]
        out = torch.transpose(out , 1 , 2) #[b , 5 , max_count ]        


        (matched_gt_u , matched_gt_vtop , matched_gt_vbtm , matched_gt_du ,matched_gt_dvtop ,matched_gt_dvbtm) ,\
        (matched_prd_u , matched_prd_vtop ,matched_prd_vbtm , matched_prd_du ,matched_prd_dvtop , matched_prd_dvbtm,),gt_idxs  = match_gt(pack_gt , out )        

        losses = cal_loss(                
            (matched_prd_u , matched_prd_vtop ,matched_prd_vbtm , matched_prd_du ,matched_prd_dvtop , matched_prd_dvbtm,) ,         
            (matched_gt_u , matched_gt_vtop , matched_gt_vbtm , matched_gt_du ,matched_gt_dvtop ,matched_gt_dvbtm) ,
            gt_idxs
        )        
        it_loss = sum(l for l in losses.values())              

        for k, v in losses.items():    
            writer.add_scalars('loss/'+k ,{'train':v.item()} , it_count + ep_count*len(dataloader))
        
        opt.zero_grad()
        it_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, norm_type='inf')
        opt.step()

        if((ep_count % MAX_LOG_GAP==0 ) and (it_count < 10)):            
            train_save_path = create_folder( os.path.join(log_folder , f"train_{ep_count}"))
            with torch.no_grad():                
                u,vt,vb,scores =  decode((matched_prd_u ,matched_prd_vtop ,matched_prd_vbtm , matched_prd_du ,matched_prd_dvtop , matched_prd_dvbtm), CONFIDENCE_THRESHOLD )
                plt_imgs = visualize_2d( 
                    u,
                    vt ,
                    vb , 
                    data['image'],
                    matched_prd_u,
                    #None,#u_grad
                    "Trainging Prediction",
                    True,
                    save_path=train_save_path
                
                )                
                u,vt,vb,scores =  decode((matched_gt_u , matched_gt_vtop , matched_gt_vbtm , matched_gt_du ,matched_gt_dvtop ,matched_gt_dvbtm),0.5 )
                plt_imgs = visualize_2d( 
                    u,
                    vt ,
                    vb , 
                    data['image'],
                    #None , #u_grad
                    data['u_grad'] ,
                    "GT",
                    save_path= train_save_path
                )
        it_count+=1        


def eval_loop( model , dataloader , ep_count ):  
    it_count    =0
    model.eval()

    pr_helper = PR_Eval_Helper(writer=writer ,ep= ep_count)
    #all_max_iou_pre_reg=[]
    #gt_count=0

    for data in tqdm(dataloader , desc="ep "+str(ep_count) , ):    
                
        for k, v in data.items():    
            data[k]=data[k].to(device)            
        
        pack_gt = (data['u'] , data['v_top'] , data['v_btm'] , data['du'] , data['dv_top'] , data['dv_btm'] , data['u_grad']  )        
        pack_gt = torch.cat(pack_gt , 1)
        b, _ = pack_gt.shape
        pack_gt = pack_gt.reshape((b,7,-1))
        pack_gt = encode(pack_gt)

        out = predict(data , model)   #[b , max_count , 6 ]
        out = torch.transpose(out , 1 , 2) #[b , 6 , max_count ]
        pack_out = (out[:,0] , out[:,1],out[:,2],out[:,3],out[:,4],out[:,5])

        # ====== Eval Loss ======
        (matched_gt_u , matched_gt_vtop , matched_gt_vbtm , matched_gt_du ,matched_gt_dvtop ,matched_gt_dvbtm) ,\
        (matched_prd_u , matched_prd_vtop ,matched_prd_vbtm , matched_prd_du ,matched_prd_dvtop , matched_prd_dvbtm,),gt_idxs  = match_gt(pack_gt , out )

        losses = cal_loss(                
            (matched_prd_u , matched_prd_vtop ,matched_prd_vbtm , matched_prd_du ,matched_prd_dvtop , matched_prd_dvbtm,) ,         
            (matched_gt_u , matched_gt_vtop , matched_gt_vbtm , matched_gt_du ,matched_gt_dvtop ,matched_gt_dvbtm) ,
            gt_idxs
        )        
        it_loss = sum(l for l in losses.values())              

        for k, v in losses.items():    
            writer.add_scalars('loss/'+k ,{'test' :v.item()} ,it_count + ep_count*len(dataloader))


        pred_u,pred_vt,pred_vb , pred_scores =  decode(pack_out , CONFIDENCE_THRESHOLD , True)                    
        gt_u,gt_vt,gt_vb,gt_u_grad =  decode((matched_gt_u , matched_gt_vtop , matched_gt_vbtm , matched_gt_du ,matched_gt_dvtop ,matched_gt_dvbtm) , 0.5 , True)
        
        pred_poly = uv_to_distorted_box(pred_u,pred_vt,pred_vb)
        gt_poly = uv_to_distorted_box(gt_u,gt_vt,gt_vb)
      
        # ========= PR Curve ===========
        #for pred , gt ,score in zip(pred_bboxes , gt_bboxes , scores  ): # each image                                    
        for pred , gt  in zip(pred_poly , gt_poly ): # each image                                    
            pr_helper.eval_batch_pr(pred , gt , None , ep_count)

        #if((ep_count % MAX_LOG_GAP==0 ) and (it_count < MAX_LOG_IT_COUNT)):    
                
        if(it_count < MAX_LOG_IT_COUNT):        
            test_save_path = create_folder( os.path.join(log_folder , f"test_{ep_count}"))
            plt_imgs = visualize_2d( 
                pred_u,
                pred_vt ,
                pred_vb , 
                data['image'],
                #None,#u_grad,
                out[:,0],                
                "inf",
                True,
                pred_poly,
                save_path=test_save_path

            )

            plt_imgs = visualize_2d( 
                gt_u,
                gt_vt ,
                gt_vb , 
                data['image'],
                #None,
                data['u_grad'],
                "gt",
                False,
                gt_poly,
                save_path=test_save_path
            )
         
        it_count +=1

    p , r ,auc = pr_helper.get_all_pr()
    #pr_helper.write_tensorboard()
    ap_50 = pr_helper.final_result_dict[1]['ap']   

    #return p , r ,auc
    return ap_50

if __name__ == '__main__':
	
	#======= [ SETTING ] =======
	MAX_PREDICTION_COUNT = Horizon_MAX_PREDICTION
	BATCH_SIZE=2
	C = Horizon_C
	R = Horizon_R
	CONFIDENCE_THRESHOLD = Horizon_CONFIDENCE_THRESHOLD
	DO_AUG= Horizon_AUG

	#MODEL_FOLDER =r'./output/'
	MODEL_FOLDER = create_folder( os.path.join(os.getcwd() , "output" , "checkpoints" ))

	#LONAD_MODEL_NAME ="n90-c0.1-r10-0912-all-final-save.pth"  
	LONAD_MODEL_NAME =""  
	TRAIN_NAME = f"__0.35loss"
	writer = SummaryWriter(TRAIN_NAME)

	TRAIN_DATASET_NAME  = "../../anno/train_visiable_20_no_cross.json"
	TEST_DATASET_NAME   = "../../anno/test_visiable_10_no_cross.json"

	#========= [ Log Setting ] ==========
	MAX_LOG_GAP = 5
	MAX_LOG_IT_COUNT = 5
	EVAL_GAP = 5
	save_auc = 0.2
	log_folder = create_folder( os.path.join(os.getcwd() , "output" , TRAIN_NAME ))
	
	#======= [ Load Model ] =======
	model = create_model(LONAD_MODEL_NAME)	
	opt = optim.Adam(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=1e-4,
			betas=(0.9, 0.999))
     
	# Train / Eval
	train_dataset = CustomDataset( f"{TRAIN_DATASET_NAME}" , use_aug = False  )  
	eval_dataset = CustomDataset( f"{TEST_DATASET_NAME}"  ,  use_aug = False ) 

	train_dataloader = DataLoader(train_dataset, BATCH_SIZE , shuffle=True, drop_last =True)
	eval_dataloader = DataLoader(eval_dataset, BATCH_SIZE , shuffle=False, drop_last =True)

	
	#ep_count = 1 if LOADED_EPOCH is None else LOADED_EPOCH+1
	#train_loop()

	#while True:
	for i in range(1,51):        
		ep_count = i
		# ======= Train EPOCH =======    
		train_loop(model , opt, dataloader= train_dataloader ,ep_count= i , log_folder= log_folder )    
		# ======= Eval EPOCH =======    
		if(ep_count % EVAL_GAP ==0):
			print("=========== EVAL =========")            
			with torch.no_grad():        
				auc = eval_loop(model , eval_dataloader,ep_count)
				
				if(auc > save_auc) :                    
					path = os.path.join(MODEL_FOLDER , f"{TRAIN_NAME}_bk_best_auc{auc}.pth")
					save_model(model, path , ep_count , auc)
					save_auc = auc
				if(ep_count % 5 == 0):
					#path = './output/'+ 'bk.pth'
					path = os.path.join(MODEL_FOLDER , f"{TRAIN_NAME}_bk.pth")
					save_model(model, path , ep_count , auc)
					