# https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
import cv2 as cv
# import required packages
import cv2
import argparse
import numpy as np
import os
import pathlib
from tqdm import tqdm

def parse_arg():
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help = 'path to input image')
    ap.add_argument('-c', '--config', required=True,
                    help = 'path to yolo config file')
    ap.add_argument('-w', '--weights', required=True,
                    help = 'path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,
                    help = 'path to text file containing class names')
    return ap.parse_args()

def load_weights(weights, config):
    # read pre-trained model and config file
    return cv2.dnn.readNet(weights, config)

def save_bounding_boxes(image, net, classes, out_file):
    width = image.shape[1]
    height = image.shape[0]
    #scale = 0.00392
    scale = 1.0 / 255.0

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (512,512), (0,0,0), False, crop=False)

    # set input blob for the network
    net.setInput(blob)
    
    # function to get the output layer names 
    # in the architecture
    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    
    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))
    
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.1
    
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #if scores.sum() > 0.5:
            #    print(classes[class_id], sorted(scores.tolist(), reverse=True)[:3])
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # go through the detections remaining
    # after nms and draw bounding box
    with open(out_file, mode='w') as f:
        for i in indices:
            i = i[0]
            box = boxes[i]
            dw = 1. / width
            dh = 1. / height
            x = (box[0] + box[2] / 2.0) * dw
            y = (box[1] + box[3] / 2.0) * dh
            w = box[2] * dw
            h = box[3] * dh
            ent = f"{class_ids[i]} {x:6f} {y:6f} {w:6f} {h:6f}\n"
            f.write(ent)

if __name__ == '__main__':
    args = parse_arg()

    net = load_weights(args.weights, args.config)

    def gen_classes():
        classes = None
        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    classes = gen_classes()
    if os.path.isdir(args.image):
        image_names = [os.path.join(args.image, im.name) for im in pathlib.Path(args.image).glob('*.png')]
    else:
        image_names = [args.image]

    for image_name in tqdm(image_names):
        # read input image
        image = cv2.imread(image_name)
        image_wo_ext = os.path.splitext(image_name)[0]
        anno_txt = image_wo_ext + ".txt"
       
        save_bounding_boxes(image, net, classes, anno_txt)
