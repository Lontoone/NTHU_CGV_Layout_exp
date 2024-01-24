from model import *
import os
import random
import cv2
import numpy as np

def inf(target_set , model):
        
    #target_set = TEST_DATASET
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    print("device" , DEVICE)
    # utils
    categories = target_set.coco.cats    
    #id2label = {k: v['name'] for k,v in categories.items()}
    id2label = {k: "door" for k,v in categories.items()}
    box_annotator = sv.BoxAnnotator()


    # select random image
    image_ids = target_set.coco.getImgIds()
    image_id = random.choice(image_ids)
    print('Image #{}'.format(image_id))

    # load image and annotatons    
    image = target_set.coco.loadImgs(image_id)[0]
    annotations = target_set.coco.imgToAnns[image_id]
    image_path = os.path.join(target_set.root, image['file_name'])
    image = cv2.imread(image_path)

    # annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
    frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    print('ground truth')    
    sv.show_frame_in_notebook(frame, (16, 16))

    # inference
    with torch.no_grad():

        # load image and predict
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            #threshold=CONFIDENCE_TRESHOLD,
            threshold=0.5,
            target_sizes=target_sizes
        )[0]
        print("results" , results)

    # annotate
    if(len(results['boxes']) > 0):
        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
        #labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        labels = [f"door {confidence:.2f}" for _, confidence, class_id, _ in detections]

        frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        print('detections')        
        sv.show_frame_in_notebook(frame, (16, 16))
    else:
        print("no result")


if  __name__ =="__main__":
    load_model_path = os.path.join(os.getcwd() , "output/checkpoints" , "d200_ep50")

    print("load_model_path" , load_model_path)
    train_file = os.path.join(os.path.dirname(os.getcwd()) , "anno" , "coco_train_200.json")
    test_file = os.path.join(os.path.dirname(os.getcwd()) , "anno" , "coco_test_10.json")
    #train_file = "coco_train_200.json"
    #test_file = "coco_test_10.json"

    train_loader , test_loader , train_dataset , test_dataset = get_loader(
        img_root_folder="E:/Projects/Datasets/Zillow_coco/train_onedrive/" ,
        try_load=False,
        train_json_path= train_file , 
        test_json_path= test_file,
        batch_size=8
            )
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DetrForObjectDetection.from_pretrained(load_model_path)
    model.to(DEVICE)
    inf(train_dataset , model)

    pass