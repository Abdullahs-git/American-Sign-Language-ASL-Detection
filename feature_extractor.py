# from_scratch_model/feature_extractor.py

import cv2
import numpy as np

def extract_hog(img, cell_size=(8, 8), block_size=(2, 2), bins=9):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180

    h, w = img.shape
    cx, cy = cell_size
    bx, by = block_size

    n_cells_x = w // cx
    n_cells_y = h // cy

    hist = np.zeros((n_cells_y, n_cells_x, bins))

    for y in range(n_cells_y):
        for x in range(n_cells_x):
            cell_mag = magnitude[y*cy:(y+1)*cy, x*cx:(x+1)*cx]
            cell_ang = angle[y*cy:(y+1)*cy, x*cx:(x+1)*cx]
            hist_cell, _ = np.histogram(cell_ang, bins=bins, range=(0,180), weights=cell_mag)
            hist[y, x] = hist_cell

    return hist.ravel()

def extract_lbp(img, P=8, R=1):
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for i in range(R, h - R):
        for j in range(R, w - R):
            center = img[i, j]
            binary = ''
            for p in range(P):
                theta = 2 * np.pi * p / P
                x = i + int(round(R * np.sin(theta)))
                y = j + int(round(R * np.cos(theta)))
                binary += '1' if img[x, y] > center else '0'
            lbp[i, j] = int(binary, 2)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    return hist

def extract_color_histogram(img, bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [bins]*3, [0,180,0,256,0,256]).flatten()
    return hist / np.sum(hist)

def extract_features(image):
    image = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat = extract_hog(gray)
    lbp_feat = extract_lbp(gray)
    color_feat = extract_color_histogram(image)
    return np.concatenate([hog_feat, lbp_feat, color_feat])
