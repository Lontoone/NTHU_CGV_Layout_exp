import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
current_file = os.getcwd()
os.chdir(dname)

import torch
import sys
print(os.getcwd())
sys.path.append('../../')
from config import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
os.chdir(current_file)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict (data , net):    
    imgs = data['image'].to(device)        
    out = net(imgs)    
    return out

def fig_to_img(fig):    
    img = np.asarray(fig.canvas.buffer_rgba())
    return img

def save_model(net, path , epoch =0 , ap = 0):
    state_dict = {
        #'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'use_rnn': net.use_rnn,
        },
        'state_dict': net.state_dict(),
        'epoch':epoch,
        'ap':ap
    }
    torch.save(state_dict, path)
# 輸出 [b , 5 , max_door_count]
#   Note: [b,0]為x座標，非原本的u_grad
def filter_peak_data(predict , min_threshold=-1):
    u  , v_top ,v_btm , d_top , d_btm  = predict

    b,n, _, w  = u.shape

    result = torch.zeros((b,5,n)).to(device)
    u = u .reshape(b,n,-1)
    v_top = v_top.reshape(b,n,-1)
    v_btm = v_btm.reshape(b,n,-1)
    d_top = d_top.reshape(b,n,-1)
    d_btm = d_btm.reshape(b,n,-1)

    peak_col_idx = torch.argmax(u,2)     

    for i in range(b):
        _t = torch.arange(Horizon_MAX_PREDICTION).to(device)
        idx = _t * w + peak_col_idx[i]             
        
        result[i,0,:]=peak_col_idx[i] / w  
        result[i,1,:]=v_top.reshape(b , -1)[i,idx]
        result[i,2,:]=v_btm.reshape(b , -1)[i,idx]
        result[i,3,:]=d_top.reshape(b , -1)[i,idx]
        result[i,4,:]=d_btm.reshape(b , -1)[i,idx]

        if(min_threshold>0):
            u_max_value_each_cha = torch.max(u[i] , 1)[0]            
            low_u_idx = torch.where(u_max_value_each_cha < min_threshold)[0]            
            result[i,0,low_u_idx] = -9999
        pass
        
        
        pass    
    return result
def visualize_2d(us, v_tops , v_btms, imgs, u_grad=None  , title =None , do_sig_u =False , polys = None ,  save_path=""):
    out_imgs=[]    
    length =  imgs.shape[0] if torch.is_tensor(imgs) else  len(imgs)        

    for i in range(length):
        if polys  is not None and u_grad is not None:            
            img =visualize_2d_single(us[i] , v_tops[i] ,v_btms[i] , imgs[i] , u_grad[i] , title ,do_sig_u , polys[i]  , save_path= f"{save_path}/{title}_{i}.jpg")
        elif u_grad is not None:            
            img =visualize_2d_single(us[i] , v_tops[i] ,v_btms[i] , imgs[i] , u_grad[i] , title ,do_sig_u ,polys, save_path= f"{save_path}/{title}_{i}.jpg" )
        else:
            img = visualize_2d_single(us[i] , v_tops[i] ,v_btms[i] , imgs[i] , None , title , do_sig_u , polys, save_path= f"{save_path}/{title}_{i}.jpg")

        out_imgs.append(img)
    return out_imgs



def visualize_2d_single(us, v_tops , v_btms, imgs, u_grad=None , title=None , do_sig_u =False , poly =None , save_path=""):
    if isinstance(us, torch.Tensor):
        us = us.cpu().detach().numpy().flatten()
        v_tops = v_tops.cpu().detach().numpy().flatten()
        v_btms = v_btms.cpu().detach().numpy().flatten()
    else:    
        us=np.array([u.cpu().detach().numpy() for u in us]).flatten()
        v_tops= np.array([u.cpu().detach().numpy().flatten() for u in v_tops]).flatten()
        v_btms=np.array([u.cpu().detach().numpy().flatten() for u in v_btms]).flatten()
    uvs=[]
    for u, v_t , v_b in zip( us , v_tops ,v_btms):   
        uvs.append( (u , v_t) )
        uvs.append( (u , v_b) )        
        
    img = imgs.permute(1,2,0).cpu().detach().numpy()
    img = np.ascontiguousarray(img)

    if(poly is not None):        
        for doors in poly:            
            for part_door in doors:            
                part_door = np.array(part_door)                
                part_door = part_door.reshape((-1 , 2)) * np.tile(np.array([1024 , 512]) , (part_door.size//2 , 1) )
                part_door = part_door.astype('int32')               
                img =  cv2.polylines(img, [part_door], True, (0,255,0), 2)
        pass
    
    h,w,c = img.shape
    img_size = [w,h]    
    for point in uvs:
        #p = np.float32(point) * img_size % img_size       # clamp to boarder     
        p = np.float32(point) * img_size         
        p = np.int32(p)        
        img = cv2.circle( img, tuple( (p[0] , p[1])), 5,(255,0,0) , thickness= -1)

    # Preview Confidence map
    if u_grad is not None:        
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=3,)
        fig.tight_layout()        
        if do_sig_u ==True:
            u_grad = torch.sigmoid( u_grad)
        dist_graph = u_grad.repeat((50,1)).cpu().detach().numpy()            
            
        ax0 = fig.add_subplot(spec[0])
        ax0.imshow(dist_graph , cmap="gray" )
        ax0.axis("off")        

        ax0 = fig.add_subplot(spec[1:])
        ax0.imshow(img , aspect='auto' )
        ax0.axis("off")        
        
        if(title is not None):
            fig.suptitle(title)
        if save_path != "":
            plt.savefig(save_path)
            plt.close() 
        '''
        plt.show()
        '''
        return fig_to_img(fig)
    else:
        #plt.title(title)
        #plt.imshow(img)
        #plt.show()
        return img
    pass


# encode for fasten training 
def encode(packed_data):
    _esp = 0.000001  # avoid door width = 0 
    packed_data[:,1] = torch.log( 0.5 - packed_data[:,1])  #v_top
    packed_data[:,2] = torch.log( packed_data[:,2] - 0.5)  #v_btm
    packed_data[:,3] = torch.log( packed_data[:,3] + _esp )  #du
    packed_data[:,4] = torch.log( 0.5 - packed_data[:,4] + _esp)  #v_top2
    packed_data[:,5] = torch.log( packed_data[:,5] - 0.5 + _esp)  #v_btm2

    zeros = torch.zeros_like(packed_data)
    is_nan = torch.isnan(packed_data)
    packed_data = torch.where(is_nan , zeros , packed_data)    
    
    return packed_data
    pass

def match_gt(gt_data , predict_data):
    
    # Filter out zero
    nonZero_idx = torch.where(gt_data[:,0,:] != 0)[0]
    nonZero_idx = torch.unique(nonZero_idx ,return_counts=True)[1]
    
    
    gt_u = gt_data[:,0,:]    
    predict_u = predict_data[:,0,:]


    b , n   = predict_u.shape

    matched_gt_u = []
    matched_gt_vtop = []
    matched_gt_vbtm = []
    matched_gt_du = []
    matched_gt_dvtop = []
    matched_gt_dvbtm = []

    matched_prd_u = []
    matched_prd_vtop = []
    matched_prd_vbtm = []
    matched_prd_du = []
    matched_prd_dvtop = []
    matched_prd_dvbtm = []
    gt_idxs=[]
    u_interval = 1 / Horizon_MAX_PREDICTION    
    for i in range(b):
        #dist_mat = distance_matrix(gt_u[i] , predict_u[i] )
        # Sort by u distance
        pos_count =nonZero_idx[i]        

        sorted_gt_u , sorted_gt_idx  = torch.sort(gt_u[i,:pos_count] , dim=0)        
        #gt_data[i,:,:pos_count] = gt_data[i,:,sorted_gt_idx]       
        gt_data[i,:-1,:pos_count] = gt_data[i,:-1,sorted_gt_idx]        # -1: dont sort u_grad
        
        u_interval_idx = (sorted_gt_u / u_interval).type(torch.long)
        gt_idxs.append(u_interval_idx)

        target = torch.zeros((6,Horizon_MAX_PREDICTION)).to(device)                                
        target[0] = gt_data[i,-1]   # u_grad
        target[0,u_interval_idx] = 1   # 分類問題
        target[1:,u_interval_idx] = gt_data[i,1:-1,:pos_count]


        matched_gt_u.append(target[0,:])
        matched_gt_vtop.append(target[1,:])
        matched_gt_vbtm.append(target[2,:])
        matched_gt_du.append(target[3,:])
        matched_gt_dvtop.append(target[4,:])
        matched_gt_dvbtm.append(target[5,:])


        matched_prd_u.append(predict_data[i,0,:])
        matched_prd_vtop.append(predict_data[i,1,:])
        matched_prd_vbtm.append(predict_data[i,2,:])
        matched_prd_du.append(predict_data[i,3,:])
        matched_prd_dvtop.append(predict_data[i,4,:])
        matched_prd_dvbtm.append(predict_data[i,5,:])
        #neg_pred_u.append(predict_data[i,0,neg_u_idx])
        
    '''
    return  matched_gt_u , matched_gt_vtop , matched_gt_vbtm ,\
            matched_gt_dtop ,matched_gt_dbtm , matched_prd_u , \
            matched_prd_vtop ,matched_prd_vbtm ,matched_prd_dtop , matched_prd_dbtm, \
            #pos_idxs,neg_idxs #neg_pred_u
    '''
    return  (matched_gt_u , matched_gt_vtop , matched_gt_vbtm , matched_gt_du ,matched_gt_dvtop ,matched_gt_dvbtm) ,\
            (matched_prd_u , matched_prd_vtop ,matched_prd_vbtm , matched_prd_du ,matched_prd_dvtop , matched_prd_dvbtm,),\
            gt_idxs
    
    pass
def get_grad_u(u ,_width = 1024 , c = Horizon_C):    
    u_len = u.shape[0]
    width = _width
    dist = torch.arange(0, width)
    #dist = dist.tile((u.shape[0],1) )            
    dist = dist.repeat((u.shape[0],1) )            
    dist = torch.abs( dist.float() - u.reshape((-1,1))*width )        
    c_dist = c ** dist              
    
    #c_dist[:u_len//2] = torch.max(c_dist[ 0::2 ] , c_dist[ 1::2 ])
    
    return c_dist

def get_grad_u_keep_batch(batch_u , pair =False , width = 1024):
    result =[]
    for u in batch_u:            
        w = u.shape[0]
        if (pair):
            u1 = get_grad_u(u[:w//2].cpu().detach() , width)
            u2 = get_grad_u(u[w//2:].cpu().detach() , width)
            result.append( torch.max(u1,u2))
        else:
            result.append(get_grad_u(u.cpu().detach() , width) )
    
    return result
def u_interval_to_real_u(u_interval , threshold = 0.25):
    
    mid_offset = 1/Horizon_MAX_PREDICTION *0.5
    line = torch.arange(Horizon_MAX_PREDICTION).to(device).float() /  float( Horizon_MAX_PREDICTION) + mid_offset    
    i = 0
    masks = []    
    for ui in u_interval:            
        zero_mask = torch.zeros_like(ui)
        one_mask = torch.ones_like(ui)
        #mask = (torch.sigmoid(ui) > threshold)
        mask = torch.where(torch.sigmoid(ui) > threshold , one_mask , zero_mask)
        masks.append(mask)
        u_interval[i] = mask * line
        i+=1
    
    return u_interval , masks

from scipy import ndimage
def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = ndimage.maximum_filter(signal, size=r, mode='wrap')    
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]

def cal_loss(pred , gt , pk_idxs):
    #b = len(gt)
    l1_loss =  torch.nn.L1Loss()
    bce = torch.nn.BCEWithLogitsLoss()
        
    gt_u = torch.cat( gt[0])    
    pred_u = torch.cat( pred[0])    
    
    #u_loss = F.binary_cross_entropy_with_logits( pred_u , gt_u)            
    u_loss = bce( pred_u , gt_u)       
    
    #non_zero_idx = torch.where(gt_u > 0)[0]  
    non_zero_idx=[]
    i=0
    for pk in pk_idxs:
        non_zero_idx.append(pk + i * Horizon_MAX_PREDICTION)
        i+=1
    non_zero_idx = torch.cat(non_zero_idx)

    # other loss
    gt_vtop = torch.cat(gt[1])[non_zero_idx]    
    pred_vtop = torch.cat(pred[1])[non_zero_idx]
    
    v_top_loss = l1_loss(pred_vtop , gt_vtop )   

    pred_vbtm   = torch.cat(pred[2])[non_zero_idx]
    gt_vbtm     = torch.cat(gt[2])[non_zero_idx]
    v_btm_loss = l1_loss(pred_vbtm , gt_vbtm )

    pred_du = torch.cat(pred[3])[non_zero_idx]    
    gt_du = torch.cat(gt[3])[non_zero_idx]
    du_loss = l1_loss(pred_du , gt_du )

    pred_dtop = torch.cat(pred[4])[non_zero_idx]    
    gt_dtop = torch.cat(gt[4])[non_zero_idx]
    d_top_loss = l1_loss(pred_dtop , gt_dtop )

    pred_dbtm = torch.cat(pred[5])[non_zero_idx]    
    gt_dbtm = torch.cat(gt[5])[non_zero_idx]
    d_btm_loss = l1_loss(pred_dbtm , gt_dbtm )

    #losses = {"u_loss":u_loss *10, "v_top":v_top_loss , "v_btm":v_btm_loss,"du":du_loss ,"d_top":d_top_loss ,"d_btm":d_btm_loss }    
    #losses = {"u_loss":u_loss , "v_top":v_top_loss  , "v_btm":v_btm_loss  }    
    losses = {"u_loss":u_loss *20, "v_top":v_top_loss , "v_btm":v_btm_loss ,"du":du_loss ,"d_top":d_top_loss ,"d_btm":d_btm_loss }    
    #losses = {"u_loss":u_loss }    
    
    return losses
    '''
    [Debug]
    _img =  gt_grad_u[0].tile((10,1)).cpu().detach().numpy()
    _img2 =  pred_grad_u[0].tile((10,1)).cpu().detach().numpy()
    plt.imshow(_img , cmap='gray')
    plt.show()
    plt.imshow(_img2,cmap='gray')
    plt.show()
    '''
    pass

def to_bbox( u_pack , vt_pack , vb_pack ):

    u_flatten  = torch.cat(u_pack)
    vt_flatten  = torch.cat(vt_pack)
    vb_flatten  = torch.cat(vb_pack)

    non_zero_idx = torch.where(u_flatten>0)[0]
    u_flatten = u_flatten[non_zero_idx]
    vt_flatten = vt_flatten[non_zero_idx].reshape(-1 , 2)
    vb_flatten = vb_flatten[non_zero_idx].reshape(-1 , 2)

    vt_flatten = torch.min(vt_flatten , 1)[0]
    vb_flatten = torch.max(vb_flatten , 1)[0]

    bboxes=[]
    for i in range(vt_flatten.shape[0]):
        x1 = u_flatten[2*i]
        x2 = u_flatten[2*i+1]
        y1 = vt_flatten[i]
        y2 = vb_flatten[i]
        bboxes.append((x1,y1,x2,y2))
    bboxes = torch.as_tensor(bboxes).reshape(-1,4)
    return bboxes 
    pass

def decode(bdata , u_thresh = 0.25 , get_raw_u = False ):    
    #b ,c ,w = datas.shape
    #for bdata in datas:
    
    u = bdata[0] 
    vt = bdata[1]
    vb = bdata[2]
    du = bdata[3]
    dvt = bdata[4]
    dvb = bdata[5]        
    pk_idxs =[]
    for u_grad in u:
        #pk_idx = find_N_peaks(u_grad.cpu().detach().numpy() , r=r , min_v = u_thresh )[0]
        u_grad = torch.sigmoid (u_grad)    
        pk_idx = find_N_peaks(u_grad.cpu().detach().numpy() , r= Horizon_R , min_v = u_thresh )        
        pk_idxs.append(pk_idx[0])
     
    
    b = len(u)    
    us =[]
    vts = []
    vbs = []
    scores= []
    for i in range(b):        
        du[i]  = torch.exp(du[i])
        pk_idx = pk_idxs[i]        
        with torch.no_grad():
            sig_u = torch.sigmoid(u[i][pk_idx])
        #scores.append(u[i][pk_idx])
        
        scores.append(sig_u)

        real_gt_u = pk_idx/Horizon_MAX_PREDICTION
        real_gt_u = torch.from_numpy(real_gt_u).to(device)
        if not get_raw_u:
            #_u = torch.hstack(( real_gt_u[i] , real_gt_u[i]+ du[i] * masks[i] )) % 1            
            #_u = torch.cat(( real_gt_u[i].float() , real_gt_u[i].float()+ du[i].float() * masks[i].float() ),0) % 1
            _u = torch.cat(( real_gt_u.float() , real_gt_u.float()  + du[i][pk_idx].float()   ),0) % 1
        else:
            #_u = torch.hstack(( real_gt_u[i] , real_gt_u[i]+ du[i] * masks[i] ))
            #_u = torch.cat(( real_gt_u[i].float() , real_gt_u[i].float()+ du[i].float() * masks[i].float() ),0)         
            _u = torch.cat(( real_gt_u .float() , real_gt_u.float()  + du[i][pk_idx].float()   ),0)

        vt1 = 0.5 - torch.exp(vt[i][pk_idx]) 
        vt2 = 0.5 - torch.exp(dvt[i][pk_idx]) 

        vb1 = torch.exp(vb[i][pk_idx]) +0.5
        vb2 = torch.exp(dvb[i][pk_idx]) +0.5

        #_vt = torch.hstack((vt1 , vt2 ))    
        #_vb = torch.hstack(( vb1, vb2))
        _vt = torch.cat((vt1 , vt2 ))    
        _vb = torch.cat(( vb1, vb2))
        
        us.append(_u)
        vts.append(_vt)
        vbs.append(_vb)

    #u_grads = get_grad_u_keep_batch(us , True)

    #return u,vt,vb , u_grad
    #return us ,vts,vbs , u_grads
    return us ,vts,vbs ,scores

def uv_to_xyz(u,v):
    uu = ( u*360-180) * 0.01745
    vv = ( v*180 -90) * 0.01745        
    
    # uv to 3D
    x =  np.cos(uu) * np.cos(vv)    
    y =  np.sin(uu) * np.cos(vv)
    z =  np.sin(vv)
    return x,y,z

def xyz_to_uv(x,y,z):
    theta   = np.arctan2(y,x)
    phi     = np.arcsin(z/(np.sqrt(x**2 +y**2+z**2)))

    theta   = (theta/ 0.01745 +180)/360
    phi     = (phi/0.01745 + 90)/180
    return theta,phi
    
def interplate_uv(u,v , count = 20):
    xs,ys,zs =[],[],[]

    for uu,vv in zip( u, v ):                    
        x,y,z = uv_to_xyz(uu,vv)
        xs.append(x)
        ys.append(y)
        zs.append(z)

   
    intp_x  = np.linspace(xs[0] , xs[1] , num=count )
    intp_y  = np.linspace(ys[0] , ys[1] , num=count )
    intp_z  = np.linspace(zs[0] , zs[1], num=count )

    # 3D to uv
    thetas  =[]
    phis    =[]
    for x,y,z in zip (intp_x, intp_y, intp_z):
        theta , phi = xyz_to_uv(x,y,z)
        thetas.append(theta)
        phis.append(phi)
    return thetas , phis
'''
def to_distorted_box(u,vt,vb , image = None  ,seg_cnt = None):

    canvas = np.zeros((512,1024,3)) if image is None else image    
    polys_per_img = []  #這張image的門，每個門可能有數個部位
    
    for _u , _vt , _bv  in zip(u, vt , vb):
        cross_idx = -1        
        previous_x = 0        
        seg_count =max(int((_u[1] -_u[0])*1024)//10 , 5) if seg_cnt is None else seg_cnt
        
        # Upper line
        all_points = [[None]*seg_count*2][0]
        thetas , phis= interplate_uv(_u,_vt , seg_count)
        i=0
        for t, p in zip(thetas , phis):            
            canvas = cv2.circle(canvas , (int(t*1024) , int(p *512)) , 3 , (255,0,0) ,-1 )            

            if(t< previous_x):
                cross_idx = i 
            
            previous_x = t
            all_points[i] = [t , p]

            i+=1

        # Bottom line
        thetas , phis= interplate_uv(_u,_bv , seg_count)        
        i=1
        for t, p in zip(thetas , phis):
            canvas = cv2.circle(canvas , (int(t*1024) , int(p *512)) , 3 , (255,0,0) ,-1 )            
            all_points[len(all_points)- i] = [t , p]
            i+=1
        
        # check cross      
        if(cross_idx >0):
            left_start_top = [0 ,all_points[cross_idx][1] ]
            left_start_btm = [0 ,all_points[seg_count + cross_idx][1] ]
            
            right_mid_top = [1 ,all_points[ cross_idx][1] ]
            right_mid_btm = [1 ,all_points[seg_count + cross_idx][1] ]

            part_right = all_points[:cross_idx] +[right_mid_top] +[right_mid_btm]+ all_points[seg_count + (seg_count - cross_idx):]
            #part_left = all_points[cross_idx: seg_count + (seg_count - cross_idx)]             
            part_left = [left_start_top] + \
                all_points[cross_idx: seg_count + (seg_count - cross_idx)] + [left_start_btm]            

            # ============= Clipping ===============
            
            part_left = np.array(part_left)
            part_right = np.array(part_right)
            right_min = part_right[0][0]
            right_clip_idx = np.where(part_right.flatten()[::2]<right_min)[0]

            if (right_clip_idx.size > 0):
                part_right[right_clip_idx,0]=1+0.00001

            left_max = part_left[len(part_left)//2][0] 
            left_clip_idx = np.where(part_left.flatten()[::2] > left_max)[0]

            if (left_clip_idx.size > 0):
                part_left[left_clip_idx,0]=left_max+0.0001
            # ============= Clipping ===============

            polys = [part_left , part_right]

        else:
            polys = [all_points]

        polys_per_img.append(polys)
   
    return polys_per_img
'''

def rearng(x):    
    half_idx = len(x)//2
    x1= x[:half_idx]
    x2= x[half_idx:]
    if(isinstance(x , np.ndarray)):
        arr = np.zeros_like(x)
    else:
        arr = torch.zeros_like(x)
    arr[::2] = x1    
    arr[1::2] = x2      

    return arr
def rearrange_decoded(u,vt,vb):    
    us ,vts ,vbs= [],[],[]
    for batch_u , batch_vt , batch_vb in zip(u,vt,vb):   
        if(isinstance(batch_u , np.ndarray)):
            ru = rearng(batch_u)        
            rvt = rearng(batch_vt)
            rvb = rearng(batch_vb)
        else:
            ru = rearng(batch_u.detach().cpu().numpy())        
            rvt = rearng(batch_vt.detach().cpu().numpy())
            rvb = rearng(batch_vb.detach().cpu().numpy())

        us.append(ru)
        vts.append(rvt)
        vbs.append(rvb)
    
    return us,vts,vbs

'''
boxu = [np.array([0.9556, 0.9921])]
boxvt = [np.array([0.3810, 0.4109])]
boxvb = [np.array([0.7335, 0.6852])]
'''
#Cross Image Set
boxu = [np.array([0.7444, 1.0161])]
boxvt = [np.array([0.2220, 0.2190])]
boxvb = [np.array([0.8964, 0.8982])]
'''

# Multi door
boxu = [np.array([0.8111, 0.3778, 1.2295, 0.6528])]
boxvt = [np.array([0.4642, 0.2425, 0.4658, 0.2188])]
boxvb = [np.array([0.5735, 0.8770, 0.5703, 0.8927])]
'''

gt_boxu = [ np.array([0.7111, 0.3578, 1.1095, 0.6528])]
gt_boxvt =[ np.array([0.4642, 0.2425, 0.4658, 0.2188])]
gt_boxvb =[ np.array([0.5735, 0.8770, 0.5703, 0.8927])]

'''
boxu = [np.array([0.1222, 0.6778, 0.7111, 0.1364, 0.6922, 0.7348])]
boxvt = [np.array([0.4698, 0.4242, 0.4627, 0.4723, 0.4376, 0.4620])]
boxvb = [np.array([0.5649, 0.6636, 0.5813, 0.5593, 0.6373, 0.5829])]
'''
boxu , boxvt , boxvb = rearrange_decoded(boxu , boxvt, boxvb)
gt_boxu , gt_boxvt , gt_boxvb = rearrange_decoded(gt_boxu , gt_boxvt, gt_boxvb)
from shapely.validation import make_valid,explain_validity
from shapely.geometry import Polygon

#==============================================
# Note :     Shapely will works most of the time, but it sometimes getting invaild shape errors.
#            So we use pixel level IoU instead of Shapely.
#==============================================

def cal_poly_iou(poly_a , poly_b):
    
    if( len(poly_a) ==1 and len(poly_b) ==1): #比對的兩扇門都沒有跨畫面
        a_pg = Polygon(poly_a[0])
        b_pg = Polygon(poly_b[0])
        
        a_pg = a_pg.buffer(0)
        a_pg = a_pg.simplify(0.0001 ,preserve_topology=False)
        b_pg = b_pg.buffer(0)
        b_pg = b_pg.simplify(0.0001 , preserve_topology=False)

        poly_intersection   = a_pg.intersection(b_pg)
        poly_union          = a_pg.union(b_pg)
        if( poly_union.area== 0):
            iou=0
        else :
            iou                 = poly_intersection.area / poly_union.area        
        return iou
        pass

    else:
        iou_matrix = np.zeros((len(poly_a) , len(poly_b)))      
        for i , a_points in enumerate( poly_a):
            a_pg = Polygon(a_points)
            a_pg.buffer(0.0001)
            a_pg = a_pg.simplify(0.001 ,preserve_topology=False)

            if(not a_pg.is_valid):                
                a_pg = make_valid(a_pg)
                print("a",a_pg.is_valid , explain_validity(a_pg))

            for j , b_points in enumerate( poly_b):
                b_pg = Polygon(b_points)
                b_pg = b_pg.buffer(0)
                b_pg = b_pg.simplify(0.001 , preserve_topology=False)

                if(not b_pg.is_valid):
                    b_pg = make_valid(b_pg)
                    print("b" , b_pg.is_valid , explain_validity(b_pg))
                
                poly_intersection   = a_pg.intersection(b_pg)
                poly_union          = a_pg.union(b_pg)
                if( poly_union.area== 0):
                    iou =0
                else:                
                    iou                 = poly_intersection.area / poly_union.area
                iou_matrix[i][j] =  np.float32( iou)
                #print("iou    " ,iou    )
        total_iou = np.sum(iou_matrix)/2
        #print("iou_matrix" , iou_matrix)
        #print("total iou" , total_iou)

        return total_iou
    pass

def get_iou_matrix_distored(gt , pred):    
    iou_matrix = np.zeros((len(gt) , len(pred)))        
    for i , _gt in  enumerate(gt):        
        for j , _pred in enumerate(pred):            
            iou  = cal_poly_iou(_gt, _pred)
            iou_matrix[i][j] = np.float32( iou)
    
    return iou_matrix

'''
for batched_uvv in zip(boxu , boxvt , boxvb ):  #each image in batch    
    u = batched_uvv[0].reshape(-1,2)
    vt = batched_uvv[1].reshape(-1,2)
    vb = batched_uvv[2].reshape(-1,2)    

    polys_point = to_distorted_box(u , vt , vb)

    u   = gt_boxu[0].reshape(-1,2)
    vt  = gt_boxvt[0].reshape(-1,2)
    vb  = gt_boxvb[0].reshape(-1,2)    

    gt_polys_point = to_distorted_box(u , vt , vb,seg_cnt=5)
    iou_matrix = get_iou_matrix_distored(gt_polys_point , polys_point)
    

    


def uv_to_distorted_box(u,vt,vb):
    polys =[]
    _boxu , _boxvt , _boxvb = rearrange_decoded(u, vt , vb)
    for bu,bvt,bvb  in zip(_boxu , _boxvt , _boxvb ):    
        _u = bu.reshape(-1,2)
        _vt = bvt.reshape(-1,2)
        _vb = bvb.reshape(-1,2)    

        p =to_distorted_box(_u , _vt , _vb )
        polys.append(p)
        
    return polys
'''