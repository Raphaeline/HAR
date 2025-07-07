import subprocess
import os
import requests
import tensorflow as tf
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

modelDir = "models"
modelFileName = "PointNetHAR.h5"
modelPath = os.path.join(modelDir, modelFileName)

modelUrl = "https://huggingface.co/Raphaeline/PointNetHAR/resolve/main/PointNetHAR.h5"

if not os.path.exists(modelPath):
    print("Model belum ada. Mendownload dari Hugging Face...")

    # Buat folder jika belum ada
    os.makedirs(modelDir, exist_ok=True)

    response = requests.get(modelUrl)
    with open(modelPath, "wb") as file:
        file.write(response.content)

    print(f"Model berhasil didownload dan disimpan di: {modelPath}")
else:
    print(f"Model sudah tersedia di: {modelPath}")

if __name__ == '__main__':
    subprocess.run(["python", "gui/gui_main.py"])
