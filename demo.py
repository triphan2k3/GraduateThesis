from LossFunction import UntargetedLoss
from POPOP import POPOP
from utils import NonOverLappingFilter, saveImg
# from ImportantPixels import importantPixels

import os
import random
import copy

from PIL import Image
import numpy as np
import time
# import matplotlib.pyplot as plt

from ultralytics import YOLO
from models import YOLODetection

import streamlit as st

import argparse

if __name__ == "__main__": 
    yolo = YOLO("yolov8n.pt")
    model = YOLODetection(yolo)   

    # UI
    max_generations = st.sidebar.slider('Maximum generations:', 25, 100, 25, 25)
    population_size = st.sidebar.slider("Population size:", 8, 64, 16, 8)
    max_modified = st.sidebar.slider("Percentage of maximum number of modified pixels:", 0.005, 0.05, 0.01, 0.005)
    pc = st.sidebar.slider("Crossover rate:", 0.1, 0.9, 0.4, 0.1)
    pm = st.sidebar.slider("Mutation rate:", 0.0, 1.0, 0.1, 0.1)
    pr0 = st.sidebar.slider("Probability of generating 0:", 0.1, 0.9, 0.3, 0.1)
    perturbation_range = st.sidebar.slider("Perturbations range:", 1.0, 3.0, 2.0, 0.5)
    early_stop = st.sidebar.checkbox("Exhaust budget", True)
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    img_file_buffer = st.file_uploader("Upload an image")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        image.save("origin.jpg")
        img_array = np.array(image) # if you want to pass it to OpenCV
        st.image(Image.open("origin.jpg"))

    if os.path.isfile("origin.jpg"): 
        button = st.button("Attack!")
    
        if button:
            # Processing
            random.seed(0)
            np.random.seed(0)

            x_test = np.array(image) / 255

            data = model(np.array(image))

            if len(data) == 0:
                model.saveRes(np.array(image), os.path.join("DemoSaved", "CleanImgRes.jpg"))
                clean_img = Image.open(os.path.join("DemoSaved", "CleanImgRes.jpg"))

                with st.container():
                    cols = st.columns(2)
                    with cols[0]:
                        st.image(clean_img, caption="Detection result on clean image")
                    with cols[1]:
                        st.info("The model cannot detect any object in this image.")
            else:
                if not os.path.exists("DemoSaved"):
                    os.makedirs("DemoSaved")

                model.saveRes(np.array(image), os.path.join("DemoSaved", "CleanImgRes.jpg"))

                boxes = np.array([(data[i][:4]).astype(np.int64) for i in range(len(data))]) # top-left, bottom-right
                cls = np.array([int(data[i][5]) for i in range(len(data))])

                loss = UntargetedLoss(model, cls, boxes)

                params = {
                  'image': x_test,
                  'boxes': boxes,
                  'loss': loss,
                  'population_size': population_size,
                  'pc': pc,
                  'pm': pm,
                  'pr0': pr0,
                  'perturbation_range': perturbation_range,
                  'max_modified': max_modified,
                  'tournament_size': 4,
                  'max_archive_size': 1000,
                  'elite_prob': 0.5,
                  'early_stop': 1 if early_stop else 0,
                #   'important_pixels': importantPixels_list
                }

                attack = POPOP(params)
                res = attack.run(max_generations)
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

                np.save(os.path.join("DemoSaved", 'result.npy'), result, allow_pickle = True)

                saveImg(model, x_test, res.pos, res.values, params['boxes'], os.path.join("DemoSaved", 'BestAttackRes.jpg'))

                clean_img = Image.open(os.path.join("DemoSaved", "CleanImgRes.jpg"))
                perturbed_img = Image.open(os.path.join("DemoSaved", 'BestAttackRes.jpg'))

                success = "Attack is SUCCESS" if res.is_adv else "Attack is FAILED"

                with st.container():
                    cols = st.columns(2)
                    with cols[0]:
                        st.image(clean_img, caption="Detection result on clean image")
                    with cols[1]:
                        st.image(perturbed_img, caption=f"Detection result on perturbed image ({success})")
