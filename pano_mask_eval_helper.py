from shapely.geometry import Polygon
import torchvision.ops
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import json

#from PE_helper import *

def process_nested_polygon_iou(gt,pred):

	total_union = None	
	#=========== Get total Union ================
	for _gt_part in gt:
		_gt_part = np.array(_gt_part, dtype=np.float32).reshape(-1,2)		
		polygon_gt = Polygon ( _gt_part)	 	
		polygon_gt = polygon_gt.buffer(0)
		#print("polygon_gt" , polygon_gt)
		
		if(total_union is None and not polygon_gt.is_empty):
			total_union = polygon_gt
		elif(not polygon_gt.is_empty):
			total_union = total_union.union(polygon_gt)

		for _pred_part in pred:						
			_pred_part = np.array(_pred_part, dtype=np.float32).reshape(-1,2)			
			polygon_pred = Polygon( _pred_part)  
			polygon_pred = polygon_pred.buffer(0)	
			
			if( not polygon_pred.is_empty and not polygon_pred.is_empty ):
				total_union		  = total_union.union(polygon_pred)
	#print("total_union.area",total_union.area)

	#=========== Get Total Intersection ================
	total_intersect = None

	for _gt_part in gt:
		_gt_part = np.array(_gt_part, dtype=np.float32).reshape(-1,2)		
		polygon_gt = Polygon ( _gt_part)	 	
		polygon_gt = polygon_gt.buffer(0)		
		for _pred_part in pred:						
			_pred_part = np.array(_pred_part, dtype=np.float32).reshape(-1,2)			
			polygon_pred = Polygon( _pred_part)  
			polygon_pred = polygon_pred.buffer(0)

			if( total_intersect is None ):
				total_intersect   = polygon_pred.intersection(polygon_gt)
			elif( total_intersect is not None ):
				merge =polygon_pred.intersection(polygon_gt)
				#print("merge" , merge)
				if( not polygon_gt.is_empty and not polygon_pred.is_empty):
					total_intersect   = total_intersect.union(merge )
			'''
			elif( not polygon_gt.is_empty and not polygon_pred.is_empty ): 
				total_intersect   = polygon_pred.intersection(polygon_gt)
			print("total_intersect" , total_intersect.area)
			'''
	
	#print("total_intersect.area" , total_intersect.area )
	#print("total_union.area" ,total_union.area )
	iou = total_intersect.area / total_union.area
	return iou


def split_cross_boundary_bbox (bbox , w = 2048 ):
	results= []
	bbox = bbox.flatten()
	max_gt = np.max(bbox.flatten())	
	if(max_gt>w):
		results.append(np.array([bbox[0]  ,bbox[1]  , w , bbox[1] , w , bbox[5] , bbox[0] , bbox[5]]))  
		results.append(np.array([0  ,bbox[1]  , bbox[2]%w , bbox[1] , bbox[2]%w , bbox[5] , 0,bbox[5]]))  
	else:
		results = [bbox]
	pass
	#results = np.array(results)
	return results
def xyxy_to_bbox_polygon(xyxy):
	return np.array([xyxy[0] ,xyxy[1] , xyxy[2],xyxy[1] , xyxy[2],xyxy[3] , xyxy[0] , xyxy[3]  ]).reshape(-1,2)

def get_polygone_iou_matrix(gts,preds):
	# row: prediction
	# col: gt
	#print("iou--gts" 	, gts)
	#print("iou--preds" 	, preds)

	iou_matrix = np.zeros((len(gts) , len(preds)))		
	
	for i , gt in enumerate(gts):				
		for j , pred in enumerate(preds):			
			iou = process_nested_polygon_iou(gt ,pred)
			'''
			try:											
				iou = process_nested_polygon_iou(gt ,pred)			
			except:
				print("-------------------IOU ERROR-------------------- : Return 0")
				iou = 0
			'''
			iou_matrix[i][j] = np.float32( iou)
			pass	   
	return iou_matrix

class PR_Eval_Helper():	
	def __init__(self, writer=None , ep=0 , log_folder ="./" , eval_type = "bbox" , file_title ="eval_result",iou_thresh = [0.05,0.5,0.75]):
		self.iou_thresh = iou_thresh
		self.gt_count=0
		self.all_iou=0
		self.results_per_batch = [{"tp":[],"fp":[],"scores":[]} for _ in self.iou_thresh]
		self.writer = writer
		self.ep = ep
		self.log_folder =log_folder
		self.file_title = file_title
		#self.log_json_path =os.path.join(log_folder , file_title)

		if not os.path.exists(self.log_folder):
			os.makedirs(self.log_folder)
	
	def get_iou_matrix(self, gts , preds ):
		iou_matrix = np.zeros((len(gts) , len(preds)))		

		for i , gt in enumerate(gts):						
			for j , pred in enumerate(preds):	
				intersect = np.count_nonzero(gt * pred)
				union = np.count_nonzero((gt + pred)>0 )
				if(union ==0):
					iou =0
				else:
					iou =  intersect / union
				pass
				iou_matrix[i][j] = iou
		pass
		return iou_matrix
	
	def eval_batch_pr( self, predict_result , gt , score ,img=None ,  debug_each_batch= False):		
		'''
		#依照score排序				
		'''
		self.scores = score
		self.predict_result = predict_result
		self.gt = gt
		
		self.gt_count+= len(gt)	 
		iou_matrix = self.get_iou_matrix(self.gt, self.predict_result)
		
		#iou_matrix = get_iou_matrix_distored(gt,predict_result)
		#print("iou_matrix" , iou_matrix)
		worst_iou = 0
		pred_count = len(self.predict_result)
		_debug_tp_list=[]
		_debug_fp_list=[]
		_debug_matched_gt=[]
		
		
		for i,iou_thersh in enumerate( self.iou_thresh):
			#print("iou_thersh", iou_thersh)
			#if self.predict_result.shape[0] > 1:
			if pred_count >= 1:
				iou_thresh_mask = np.where(iou_matrix >= iou_thersh , 1 , 0 )				
				masked_iou_matrix = iou_matrix * iou_thresh_mask				
				each_gt_best_iou_idx = np.argmax(masked_iou_matrix , axis=1)				
				
				mono_gt = np.zeros_like(iou_matrix)
				if(iou_matrix.size>0):
					# 過濾掉同個gt box有多個符合的box (同row trim)，只保留最好的
					mono_gt[[i for i in range(len(mono_gt))] , [each_gt_best_iou_idx]] = iou_matrix[[i for i in range(len(mono_gt))] , [each_gt_best_iou_idx]]
					
					# 過濾掉同個pred box有多個符合的gt box (同column trim)，只保留最好的
					mono_pred = np.zeros_like(iou_matrix)
					each_pred_best_iou_idx = np.argmax(masked_iou_matrix , axis=0)   
					mono_pred[[each_pred_best_iou_idx] , [i for i in range(mono_gt.shape[1])] ] = mono_gt[ [each_pred_best_iou_idx] , [i for i in range(mono_gt.shape[1])]]
				
					mono_gt = mono_pred
				# 紀錄miou
				if(i==0):
					self.all_iou += np.sum(mono_gt)

				mono_gt *= iou_thresh_mask  # 沒達到threshold的設為0				
				#找到單列最好的 (?)
				pred_filtered_mask = np.zeros(pred_count)	  
				_temp_matched_gt = []
				for row_gt in mono_gt:
					_max_idx = np.argmax(row_gt)
					if( row_gt[_max_idx]>0 ):
						pred_filtered_mask[_max_idx]=1																
						_temp_matched_gt.append(_max_idx)
					else:
						_temp_matched_gt.append(-1)
					
				_debug_matched_gt.append(_temp_matched_gt)
				
				tp_list = np.where(pred_filtered_mask > 0 , 1 , 0 )   
				fp_list = np.where(pred_filtered_mask ==0 , 1 , 0 )   
				
				#最差的那把   
				if(mono_gt.sum()==0):
					worst_idx = -1
				else:
					worst_idx = np.argmin(mono_gt.flatten())				
					worst_iou = mono_gt.flatten()[worst_idx]	
				
				#print("batch append self.scores" , self.scores)
				self.results_per_batch[i]['tp'].append(tp_list.flatten())
				self.results_per_batch[i]['fp'].append(fp_list.flatten())
				self.results_per_batch[i]['scores'].append(self.scores)	

				#print("batch append tp" , tp_list.shape , tp_list , " len " , len(self.results_per_batch[i]['tp'] ))
				#print("------------------")
				_debug_tp_list.append(np.sum(tp_list).item())
				_debug_fp_list.append(np.sum(fp_list).item())
				
			else:
				batch_ap = 0
		if(not debug_each_batch):
			return worst_iou
		else:
			return worst_iou , _debug_tp_list , _debug_fp_list , len(gt) ,_debug_matched_gt	
			
		#return batch_pr , batch_rc , batch_ap
	
	def list_to_pr_auc(self, tp_list , fp_list , gt_count):
		tp = np.cumsum(tp_list)	 
		fp = np.cumsum(fp_list) 
		all_prediction = tp+fp
		precision = tp / all_prediction
		recall = tp / gt_count			
		self.all_prediction = precision

		precision = np.insert(precision , 0 ,1)
		recall = np.insert(recall , 0 , 0)

		auc = metrics.auc(recall,precision)
		return precision , recall,auc
	
	def get_all_pr(self):
		# combine each batch result and sort by scores
		self.final_result_dict =  [{} for _ in self.iou_thresh]		

		for i , thresh in enumerate(self.iou_thresh):			
			all_tp = np.concatenate(self.results_per_batch[i]['tp'][:])
			all_fp = np.concatenate(self.results_per_batch[i]['fp'][:])

			sum_tp = np.sum(all_tp)
			sum_fp = np.sum(all_fp)

			recall_rate = sum_tp / self.gt_count
			precision_rate = sum_tp / (sum_tp + sum_fp)
				
			if self.gt_count >0 :						
				precision , recall , auc = self.list_to_pr_auc(all_tp , all_fp , self.gt_count)
			else:
				precision = []
				recall=[]
				auc=0
				recall_rate =0
				precision_rate=0
				
			self.final_result_dict[i]=(
				{"iou_thresh": thresh ,
					"recall":recall,
					"precision":precision,
					"recall_rate": recall_rate,
					"precision_rate": precision_rate,
					"all_tp" : all_tp,
					"all_fp" : all_fp,
					"sum_tp" : np.sum( all_tp),
					"sum_fp" : np.sum( all_fp),
					"gt_count" : self.gt_count,
					"ap":auc} 
			)
			print(f"------------- final_result_dict---iou_threshold {thresh}-----------------")
			for key, value in self.final_result_dict[i].items():
				print(f'Key: {key}, Value: {value}')
			#print(self.final_result_dict[i])
			#print(f"ap_{thresh}",auc)
			#print("tp",sum_tp)
			#print("fp",sum_fp)
			#print(f"Precision_rate_{thresh}" , self.final_result_dict[i]['precision_rate'])
			#print(f"Recall_rate_{thresh}" , self.final_result_dict[i]['recall_rate'])
		print("gt count",self.gt_count)
		print("all_iou" , self.all_iou)
		print("mIOU" , self.all_iou/self.gt_count)
			
		if self.writer is not None:
			self.write_tensorboard()
		
		return precision,recall,auc
	
	def write_tensorboard(self, subName ="sub"):		
		writer = self.writer

		for i , thresh in enumerate(self.iou_thresh):			
			#prcs = np.ascontiguousarray(self.final_result_dict[i]["precision"] )
			#recs = np.ascontiguousarray(self.final_result_dict[i]["recall"] )
			prcs = self.final_result_dict[i]["precision"] 
			recs = self.final_result_dict[i]["recall"]			 
			
			step = 0
			for prc, rec in zip(prcs , recs ):
				if(self.writer is not None):
					#writer.add_scalar(f"{subName}/Precision_{thresh}-ep{self.ep}" , prc , step)
					#writer.add_scalar(f"{subName}/Recall_{thresh}-ep{self.ep}" , rec , step)
					#writer.add_scalar(f"{subName}/AUC_{thresh}-ep{self.ep}" ,  prc , rec )  # tensor board bug													
					writer.add_scalars(f"Eval/Precision_{thresh}",{f"ep{self.ep}":prc}  , step)
					writer.add_scalars(f"Eval/Recall_{thresh}" ,{f"ep{self.ep}":rec}, step)
					#writer.add_scalars(f"Eval/AUC_{thresh}" ,{f"ep{self.ep}": prc},  self.ep )  # tensor board bug													
				
				step+=1

			writer.add_scalar(f"Eval/Precision_rate_{thresh}", self.final_result_dict[i]['precision_rate'] , self.ep)
			writer.add_scalar(f"Eval/Recall_rate_{thresh}" ,   self.final_result_dict[i]['recall_rate'] , self.ep)
			writer.add_scalar(f"Eval/AUC_{thresh}" ,  self.final_result_dict[i]['ap'] , self.ep )  # tensor board bug													
			writer.add_scalar(f"Eval/mIou" ,  self.all_iou/self.gt_count , self.ep )  # tensor board bug													

			
			
			plt.plot( recs ,prcs )			
			#plt.xlim([0, 1])
			#plt.ylim([0, 1])
			plt.title(f"PR_curve-{thresh} ap : {self.final_result_dict[i]['ap']}")
			plt.savefig(self.log_folder+f"/_PR_curve-{thresh}-{subName}-ep{self.ep}.jpg")
			#self.writer.add_figure(f"PR_curve-{thresh}-ep{self.ep}-{subName}.jpg" , plt.figure())
			plt.show()

		# Serializing json
		result_dict=[]
		for i , thresh in enumerate(self.iou_thresh):	  
			result_dict.append({				
				"threshold" : thresh,
				"precision" : self.final_result_dict[i]['precision_rate'], 
				"recall" : self.final_result_dict[i]['recall_rate'], 
				"mIou" : self.all_iou/self.gt_count, 
				"ap": self.final_result_dict[i]['ap'],

				"all_tp" :  self.final_result_dict[i]['all_tp'].tolist(),
				"all_fp" :  self.final_result_dict[i]['all_fp'].tolist(),
				"sum_tp" :  self.final_result_dict[i]['sum_tp'].item(),
				"sum_fp" :  self.final_result_dict[i]['sum_fp'].item(),
				"gt_count" : self.gt_count,
			})
		json_object = json.dumps(result_dict)
		
		# Writing to sample.json
		with open(os.path.join(self.log_folder , f"{self.file_title}.json"), "w") as outfile:
			outfile.write(json_object)
		
			
		self.writer.close()
		pass