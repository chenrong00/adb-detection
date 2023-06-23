from ultralytics import YOLO 
import sys
import subprocess
import torch

model_path = "C:/Users/antho/jupyter_notebooks/FYP/train_nano/weights/best.pt"
image_video_path = "C:/Users/antho/jupyter_notebooks/FYP/ash_video_001.mp4"

if __name__ == "__main__":
	torch.cuda.is_available()
	subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
	'ultralytics'])
	model = YOLO(model_path)
	model.predict(image_video_path, save=True, conf=0.4)

	