import numpy as np
from PIL import Image
from ultralytics import YOLO, checks, hub

model = YOLO("yolo26n-seg.pt")

model.train(
    data=r"C:\University\Computer vision project\roads_yolo_seg\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,     # reduce if you get out-of-memory
    device='cpu'     # set to "cpu" if no GPU
)