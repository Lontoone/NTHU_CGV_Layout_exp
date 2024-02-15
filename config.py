
#ZILLOW_DATASET_FOLDER = '/CGVLAB3/datasets/chingjia/data/data'
ZILLOW_ZIND_JSON_PATH = 'F:/THU/DoorSeg/OneDrive_2023-01-07/Door_Detection/zind/zind/zind_partition.json'
#ZILLOW_ZIND_JSON_PATH = '/CGVLAB3/datasets/chingjia/zind/zind/zind_partition.json'
ZILLOW_DATASET_FOLDER = 'F:/THU/DoorSeg/OneDrive_2023-01-07/Door_Detection/data/data'
#ZILLOW_DATASET_FOLDER = 'E:\One_Drive_nutc\OneDrive - 國立臺中科技大學\清華\Room Layout Estimation\Door Detection\data\data'
#ZILLOW_DATASET_FOLDER = 'D:\Projects\Door_Detection\data\data'

NUMBER_WORKESRS = 4

IMG_WIDTH = 1024
IMG_HEIGHT = 512

ZILLOW_TRAIN_ANNO_JSON_PATH = './anno/train_visiable_20_no_cross.json'
ZILLOW_TEST_ANNO_JSON_PATH = './anno/test_visiable_10_no_cross.json'

ZILLOW_TEST_COCO_JSON_PATH = '../anno/instances_test2017.json'
ZILLOW_TRAIN_COCO_JSON_PATH = '../anno/instances_train2017.json'

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