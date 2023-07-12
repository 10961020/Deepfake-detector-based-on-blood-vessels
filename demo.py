import cv2
import os
import time
import glob
import dlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers, metrics, models
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.ndimage import binary_dilation, binary_erosion

from utils.model_utils import ResidualAttentionNetwork
from utils.process import load_facedetector
from utils.load import get_ids, split_train_val, get_imgs_and_masks, batch, grad, list_transform_np, normalize, get_imgs


def val(batch_size=1):
    net = models.load_model(os.path.join(config['output_path'], 'network_weight.h5'))
    img = glob.glob(os.path.join(config['input_path_val'], '*.png'))
    img = [r'./face_imgs/real_02532.png']
    
    face_detector, landmark_Predictor = load_facedetector(config)

    image_next = get_imgs(img, face_detector, landmark_Predictor) # Section 3.1  Sclera Segmentation 

    value = []
    for i, (x, y) in enumerate(batch(image_next, batch_size)):
        x = list_transform_np(x)
        x = normalize(x)
        y_ = net(x)
        value.append(y)
        print('The prediction for this image is:', np.array(y_)[0][0])


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    config = {
        'input_path': r'E:\zhangtong\bloodshot_research\face_img\train',
        'input_path_val': r'E:\zhangtong\bloodshot_research\face_img\val',
        'output_path': r'E:\zhangtong\bloodshot_research\output',
        'facedetector_path': r'E:\zhangtong\bloodshot_research/shape_predictor_68_face_landmarks.dat',
    }

    val()


