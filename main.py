from LossFunction import UntargetedLoss
from POPOP_ImportantPixels import POPOP
from utils import NonOverLappingFilter, saveImg
# from ImportantPixels import importantPixels

import os
import random
import copy

from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

from ultralytics import YOLO
from models import YOLODetection

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--max_modified", type=float, default=0.01)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--pc", type=float, default=0.4)
    parser.add_argument("--pm", type=float, default=0.1)
    parser.add_argument("--pr0", type=float, default=0.3)
    parser.add_argument("--perturbation_range", type=float, default=2.0)
    parser.add_argument("--tournament_size", type=int, default=4)
    parser.add_argument("--max_archive_size", type=int, default=1000)
    parser.add_argument("--elite_prob", type=float, default=0.5)
    parser.add_argument("--max_generations", type=int, default=50)
    parser.add_argument("--early_stop", type=int, default=1, help="0 or 1")

    parser.add_argument("--save_directory", type=int)
    args = parser.parse_args()
    
    assert args.early_stop == 0 or args.early_stop == 1

    yolo = YOLO("yolov8n.pt")
    model = YOLODetection(yolo) # Change this with your model, make sure the outputs format are the same with YOLO, ***see models.py***

    random.seed(0)
    np.random.seed(0)
    PATH = args.img_path

    img_list = os.listdir(PATH)
    img_list.sort(reverse=False)
    cnt = 0
    success_cnt = 0

    for name in img_list[:len(img_list) // 2 + 1]:
        folder = name.replace(name[-4:], "")
        if not os.path.exists(folder):
            os.mkdir(folder)

        image = Image.open(os.path.join(PATH, name))

        x_test = np.array(image) / 255

        x_test = np.array(image) / 255

        data = model(np.array(image))
        if len(data) == 0:
            continue
        model.saveRes(np.array(image), os.path.join(folder, "CleanImgRes.jpg"))

        # data[i] will get the infor of the (i+1)-th highest confident object (has the (i+1)-th highest confidence score)
        boxes = np.array([(data[i][:4]).astype(np.int64) for i in range(len(data))]) # top-left, bottom-right
        cls = np.array([int(data[i][5]) for i in range(len(data))])

        ########################################################################################
        '''
        this version doesn't use NonOverLappingFilter()

        # print(f'Before filtering: box\'s shape - {boxes.shape}, cls\'s shape - {cls.shape}')

        # boxes, cls = NonOverLappingFilter(boxes, cls)

        # print(f'Before filtering: box\'s shape - {boxes.shape}, cls\'s shape - {cls.shape}')
        '''
        ########################################################################################

        # importantPixels_list = importantPixels(image, boxes)

        max_modified = args.max_modified #int(h * w * 1/ 100)
        loss = UntargetedLoss(model, cls, boxes)

        params = {
          'image': x_test,
          'boxes': boxes,
          'loss': loss,
          'population_size': args.population_size,
          'pc': args.pc,
          'pm': args.pm,
          'pr0': args.pr0,
          'perturbation_range': args.perturbation_range,
          'max_modified': max_modified,
          'tournament_size': args.tournament_size,
          'max_archive_size': args.max_archive_size,
          'elite_prob': args.elite_prob,
          'early_stop': args.early_stop,
        #   'important_pixels': importantPixels_list
        }

        attack = POPOP(params)
        res = attack.run(args.max_generations)

        result = {
            'loss': res.loss,
            'is_adv': res.is_adv,
            'pos': res.pos,
            'values': res.values,
            'l0': res.l0,
            'l2': res.l2,
            'num_perturbed': res.num_perturbed,
            'archive': [[a.loss, a.is_adv, a.pos, a.values, a.l0, a.l2, a.num_perturbed] for a in attack.archive],
            'iters': attack.iters,
            'log': attack.log
        }

        np.save(os.path.join(folder, 'result.npy'), result, allow_pickle = True)

        saveImg(x_test, res.pos, res.values, params['boxes'], os.path.join(folder, 'BestAttackRes.jpg'))

        cnt += 1
        if res.is_adv:
            success_cnt += 1

        print(f"Img: {name} - suceess: {res.is_adv} - loss: {res.loss:.2f} - l2: {res.l2:.2f} - l0: {res.l0} - num_perturbed: {res.num_perturbed} - current ASR: {success_cnt / cnt}")