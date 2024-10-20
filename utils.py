import numpy as np
from PIL import Image
import os
import random
import copy

def IoU(box1, box2):
    # print(box1, box2)
    top1 = box1[:2]
    bottom1 = box1[2:]
    top2 = box2[:2]
    bottom2 = box2[2:]

    area1 = (bottom1[0] - top1[0]) * (bottom1[1] - top1[1])
    area2 = (bottom2[0] - top2[0]) * (bottom2[1] - top2[1])

    I = max(0, min(bottom1[0], bottom2[0]) - max(top1[0], top2[0])) * max(0, min(bottom1[1], bottom2[1]) - max(top1[1], top2[1]))
    U = area1 + area2 - I
    
    return I / U

def NonOverLappingFilter(boxes, cls):
    non_overlapping_boxes = []
    non_overlapping_cls = []
    
    for box, cls in zip(boxes, cls):
        if len(non_overlapping_boxes) == 0:
            non_overlapping_boxes.append(box)
            non_overlapping_cls.append(cls)
            continue
          
        iou_scores = [IoU(b, box) for b in non_overlapping_boxes]
        if max(iou_scores) < 1e-10:
            non_overlapping_boxes.append(box)
            non_overlapping_cls.append(cls)
    
    return np.array(non_overlapping_boxes), np.array(non_overlapping_cls)

def saveImg(model, img, pos, values, boxes, path):
    perturbed_img = img.copy()

    for j, box in enumerate(boxes):
        w = abs(box[2] - box[0])
        h = abs(box[3] - box[1])

        for i in range(len(pos[j])):
            posH = min(box[1] + pos[j][i] // w, img.shape[0] - 1)
            posW = min(box[0] + pos[j][i] % w, img.shape[1] - 1)

            perturbed_img[int(posH), int(posW)] += values[j][i]

    perturbed_img = np.clip(perturbed_img, 0, 1)
    model.saveRes((perturbed_img * 255).astype(np.uint8), path)

