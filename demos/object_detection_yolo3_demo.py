import os
from PIL import Image
import matplotlib.pyplot as plt
from cortex.models.yolo.detector import YOLO, detect_video


dir_path = os.path.dirname(os.path.realpath(__file__))
cortex_base = os.path.normpath(os.path.join(dir_path, '..', 'cortex'))
coco_classes_path = os.path.join(cortex_base, 'dataset', 'coco_classes.txt')
anchors_path = os.path.join(cortex_base, 'models', 'yolo', 'yolo_anchors.txt')
model_path = os.path.join(cortex_base, 'models', 'yolo', 'yolo.h5')

image_path = '~/data/images/00000.png'
image = Image.open(image_path)

options = {"classes_path": coco_classes_path, 
           "anchors_path" : anchors_path,
           "model_path" : model_path}

detector = YOLO(options)
output = detector.detect_image(image)
plt.imshow(output)
