import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load a pre-trained model
model_2cls = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True , **{"box_nms":0.25})

# Get the number of input features for the classifier
in_features = model_2cls.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one (note: +1 because of the background class)
num_classes = 2  # 1 class (your new class) + background
model_2cls.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
