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
    real_img = glob.glob(os.path.join(config['input_path_val'], 'real_*.png'))
    style_img = glob.glob(os.path.join(config['input_path_val'], 'style_*.png'))

    face_detector, landmark_Predictor = load_facedetector(config)
    y_real = []
    y_style = []

    real_next = get_imgs(real_img, face_detector, landmark_Predictor)
    style_next = get_imgs(style_img, face_detector, landmark_Predictor)

    value = []
    for i, (x, y) in enumerate(batch(real_next, batch_size)):
        x = list_transform_np(x)
        x = normalize(x)
        y_ = net(x)
        value.append(y)

        y_real.append(np.array(y_)[0][0])

    for i, (x, y) in enumerate(batch(style_next, batch_size)):
        x = list_transform_np(x)
        x = normalize(x)
        y_ = net(x)
        value.append(y)

        y_real.append(np.array(y_)[0][0])


    y_real = np.array(y_real).reshape(-1)
    y = np.array(value).reshape(-1)
    fp, tp, thr = roc_curve(y, y_real)
    roc_auc = auc(fp, tp)


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


