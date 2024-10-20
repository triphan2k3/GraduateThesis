import numpy as np
from PIL import Image
import os
import random
import copy
import cv2

class Detection:
    def __init__(self, model):
        self.model = model

    def __call__(self, image):
        """
        image: is np.array with integer value from 0 - 255
        return:
        Need to convert output to np.array, in which:
        [
          [x11, y11, x21, y21, confidence1, class1],
          [x12, y12, x22, y22, confidence2, class2],
          ...
        ]
        """
        pass

    def saveRes(self, image, path):
        pass

class YOLODetection(Detection):
    def __init__(self, model):
        super().__init__(model)

    def __call__(self, image):
        image = Image.fromarray(image)
        image.save("image.jpg")
        results = self.model(["image.jpg"], verbose = False)

        return results[0].boxes.data.numpy()
    
    def saveRes(self, image, path):
        image = Image.fromarray(image)
        image.save("../save.jpg")
        results = self.model(["../save.jpg"], verbose = False)
        results[0].save(filename = path)

# class FasterRCNNDetection(Detection):
#     def __init__(self, model):
#         super().__init__(model)

#     def __call__(self, image):
#         image = Image.fromarray(image)
#         image.save("image.jpg")
#         im = cv2.imread("image.jpg")
#         outputs = self.model(im)
#         boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
#         labels = outputs["instances"].pred_classes.cpu().numpy()
#         scores = outputs["instances"].scores.cpu().numpy()
#         results = np.concatenate([boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)], axis = 1)
        
#         return results
    
#     def saveRes(self, image, path):
#         image = Image.fromarray(image)
#         image.save("../save.jpg")
#         im = cv2.imread("../save.jpg")
#         outputs = self.model(im)
#         v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.model.cfg.DATASETS.TRAIN[0]), scale=1.2)
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         cv2.imwrite(path, out.get_image()[:, :, ::-1])