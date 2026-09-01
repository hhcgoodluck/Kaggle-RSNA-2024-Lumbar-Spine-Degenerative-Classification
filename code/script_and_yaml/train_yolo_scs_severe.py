import sys
model_name = sys.argv[1]
FOLD = sys.argv[2]
img_size = int(sys.argv[3])
EPOCHS = int(sys.argv[4])
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import glob
os.environ['WANDB_DISABLED'] = 'true'

OD_INPUT_SIZE = img_size
BATCH_SIZE = 32

from ultralytics import YOLO
model = YOLO(model_name)
model.train(project=f"{model_name.split('.')[0]}_{OD_INPUT_SIZE}_fold{FOLD}_scs_severe", data=f"yolo_scs_fold{FOLD}_{OD_INPUT_SIZE}_severe.yaml", 
            epochs=EPOCHS, imgsz=OD_INPUT_SIZE, batch=BATCH_SIZE,save_period=-1,workers=4,
            plots=False,
           )
