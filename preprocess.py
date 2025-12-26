import os
import cv2
from filters import to_grayscale, gaussian_blur, gaussian_median_blur


"val_path contient nombreux dossiers de vidéos, chaque dossier contient des frames d'une vidéo.  "
"lire chaque frame d'une vidéo (une vidéo = un dossier), puis  "
def preprocess_frame(val_path, output_path, mode = "gaussian"):
    for video in os.listdir(val_path):
        video_path= os.path.join(val_path, video) #exm : val/Ac4002
        if not os.path.isdir(video_path): 
            continue
        save_path= os.path.join(output_path, video) # out/Ac4002
        os.makedirs(save_path, exist_ok=True)
        frames = sorted(os.listdir(video_path))
        for frame in frames:
            img_path = os.path.join(video_path, frame) # val/Ac4002/000001.jpg
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = to_grayscale(img)
            if mode == "gaussian":
                out = gaussian_blur(gray, kernel_size=(5, 5), sigma=1.0)
            elif mode == "gaussian_median":
                out = gaussian_median_blur(gray, kernel_size=(5, 5), sigma=1.0, m_ksize=5)
            else:
                raise ValueError("Invalid mode selected. Choose from 'gaussian' or 'gaussian_median'.")
            
            cv2.imwrite(os.path.join(save_path, frame), out)

        