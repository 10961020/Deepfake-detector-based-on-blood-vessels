import cv2
import time
import os
import skimage
import re
import requests
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# TODO requests
#
# folder_path = './data_style_1024xp'
# url_path = 'https://thispersondoesnotexist.com/image'
#
# num = 13811
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36'}
# while True:
#     response = requests.get(url_path, headers=headers)
#     print(os.path.join(folder_path,str(num)+'.png'))
#     with open(os.path.join(folder_path,str(num)+'.png'), 'wb') as file:
#         file.write(response.content)
#     num+=1
#     time.sleep(2)


def rotate_img(img, angle):
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img


before = r'E:\zhangtong\bloodshot_research\face_img\val_random'     # original image
after = r'E:\zhangtong\bloodshot_research\face_img\val_JPEG'        # processing save path
if not os.path.exists(after):
    os.makedirs(after)
for root, dirs, files in os.walk(before):
    for file in files:
        # TODO Median filtering
        # img = cv2.imread(os.path.join(root, file))
        # img= cv2.medianBlur(img, 3)
        # cv2.imwrite(os.path.join(after, file[:-4] + '.png'), img)

        # TODO Resize
        # img = cv2.imread(os.path.join(root, file))
        # a = 0.8
        # img = cv2.resize(img, dsize=None, fx=a, fy=a)
        # cv2.imwrite(os.path.join(after, file[:-4] + '.png'), img)

        # TODO Gaussian blur
        # img = cv2.imread(os.path.join(root, file))
        # img_blur = cv2.GaussianBlur(img, (3, 3), 0.2)
        # cv2.imwrite(os.path.join(after, file[:-4] + '.png'), img_blur)

        # TODO Gaussian noise
        # img = cv2.imread(os.path.join(root, file))
        # noisy = skimage.util.random_noise(img, mode='gaussian', var=0.4)
        # cv2.imwrite(os.path.join(after, file[:-4] + '.png'), noisy+img)

        # TODO JPEG compression
        img = cv2.imread(os.path.join(root, file))
        cv2.imwrite(os.path.join(after, file[:-4]+'.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # TODO Rotation
        # img = cv2.imread(os.path.join(root, file))
        # img = rotate_img(img, 5)
        # cv2.imwrite(os.path.join(after, file[:-4] + '.png'), img)
        pass

