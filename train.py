import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from ultralytics import YOLO
import torch

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolo26n-seg.pt')
    model.train(
        data='roadsigns.yaml',
        epochs=30,
        imgsz=640,
        batch=4,
        workers=0,
        device=device,
        cache=False,
        amp=True,
        name='roadsigns_seg',
        exist_ok=True
    )

if __name__ == '__main__':
    main()