import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

im = cv2.imread("/home/vybt/Desktop/catwalk/0af5f9503b5b72ae280c28812222c822.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# print(model_zoo.get_config_file("/home/vybt/Desktop/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml"))
# cfg.merge_from_file("/home/vybt/Desktop/densepose_rcnn_R_101_FPN_DL_WC1_s1x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("/home/vybt/Desktop/model_final_289019.pkl")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

import cv2

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow()
cv2.imshow('ee', out.get_image()[:, :, ::-1])
cv2.waitKey(0)