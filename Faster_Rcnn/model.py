import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from torchvision.models import ResNet50_Weights
#from torchvision.models import resnet50

# Load a pre-trained model
model_2cls = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True ,**{"box_nms":0.25})
#model_2cls = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None,  trainable_backbone_layers = 5,weights_backbone= 'resnet50-0676ba61.pth',**{"box_nms":0.25})

# Get the number of input features for the classifier
in_features = model_2cls.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one (note: +1 because of the background class)
num_classes = 2  # 1 class (your new class) + background
model_2cls.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
