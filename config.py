
ZILLOW_DATASET_FOLDER = '/CGVLAB3/datasets/chingjia/data/data'
ZILLOW_ZIND_JSON_PATH = '/CGVLAB3/datasets/chingjia/zind/zind/zind_partition.json'
#ZILLOW_DATASET_FOLDER = 'F:/THU/DoorSeg/OneDrive_2023-01-07/Door_Detection/data/data'
#ZILLOW_ZIND_JSON_PATH = 'F:/THU/DoorSeg/OneDrive_2023-01-07/Door_Detection/zind/zind/zind_partition.json'

NUMBER_WORKESRS = 4

IMG_WIDTH = 1024
IMG_HEIGHT = 512


#==============================================
#				  Faster RCNN
#==============================================

#==============================================
#				 Horizon Net
#==============================================
Horizon_MAX_PREDICTION = 90
Horizon_C = 0.1
Horizon_R = 10
Horizon_CONFIDENCE_THRESHOLD = 0.05
Horizon_AUG = False   # do augmentation