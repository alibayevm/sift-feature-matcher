from os.path import join, isdir
from os import makedirs
import cv2
import numpy as np

_MATCH_THRES = 0.6
_MAX_SIZE = 512*512

def circular_convolution(array, n):
    temp = array[:]
    convolved = [0 for _ in range(len(array))]
    for _ in range(n):
        for i in range(len(array)):
            left = (i-1+len(array)) % len(array)
            right = (i+1) % len(array)
            convolved[i] = (temp[left] + temp[i] + temp[right]) / 3
        temp = convolved[:]
    return convolved

def save_gaussian_scale_space(gaussians):
    if not isdir(join('outputs', 'gaussians')):
        makedirs(join('outputs', 'gaussians'))
    for i in range(len(gaussians)):
        for j in range(len(gaussians[0])):
            out_path = join('outputs', 'gaussians', 'octave_{}_level{}.jpg'.format(i+1, j))
            cv2.imwrite(out_path, gaussians[i][j]*255)

def save_difference_of_gaussians(dogs):
    if not isdir(join('outputs', 'diff_of_gaussians')):
        makedirs(join('outputs', 'diff_of_gaussians'))
    for i in range(len(dogs)):
        for j in range(len(dogs[0])):
            out_path = join('outputs', 'diff_of_gaussians', 'octave_{}_level{}.jpg'.format(i+1, j))
            cv2.imwrite(out_path, dogs[i][j]*255)

def draw_keypoints(img_name, keypoints):
    img = cv2.imread(join('images', img_name))
    for keypoint in keypoints:
        _, _, row, col = keypoints[keypoint]
        row = int(round(row))
        col = int(round(col))
        cv2.circle(img, (col, row), 4, (255, 0, 0))
    img_name = img_name.split('.')[0] + '_kp.' + img_name.split('.')[1]
    cv2.imwrite(join('outputs', img_name), img)

def match_descriptors(descriptors_one, descriptors_two):
    matches = {}
    for kp1 in descriptors_one:
        desc1 = descriptors_one[kp1]
        
        min_dist_first = float('inf')
        for kp2 in descriptors_two:
            desc2 = descriptors_two[kp2]
            if np.linalg.norm(desc1-desc2) < min_dist_first:
                min_dist_first = np.linalg.norm(desc1-desc2)
                nearest = kp2
        
        min_dist_second = float('inf')
        for kp2 in descriptors_two:
            desc2 = descriptors_two[kp2]
            if np.linalg.norm(desc1-desc2) < min_dist_second and kp2 != nearest:
                min_dist_second = np.linalg.norm(desc1-desc2)

        if min_dist_first < _MATCH_THRES * min_dist_second:
            matches[kp1] = nearest
    return matches

def draw_matches(matches, img1_pth, img2_pth):
    img1 = cv2.imread(join('images', img1_pth))
    img2 = cv2.imread(join('images', img2_pth))

    imgNames = img1_pth.replace('_resized', '').split('.')[0] + '_' + img2_pth.replace('_resized', '').split('.')[0]
    if not isdir(join('outputs', 'matches', imgNames)):
        makedirs(join('outputs', 'matches', imgNames))
    
    img = np.zeros(shape=(img1.shape[0]+img2.shape[0], img1.shape[1]+img2.shape[1], 3))
    img[:img1.shape[0], :img1.shape[1]] = img1 
    img[img1.shape[0]:, img1.shape[1]:] = img2
    all_matches = img.copy() 

    for i, kp1 in enumerate(matches):
        kp2 = matches[kp1]
        imgLine = img.copy()
        _, row1, col1, _ = kp1
        _, row2, col2, _ = kp2
        cv2.line(imgLine, (int(col1), int(row1)), (int(col2) + img1.shape[1], int(row2) + img1.shape[0]), (0, 255, 0), 1)
        cv2.line(all_matches, (int(col1), int(row1)), (int(col2) + img1.shape[1], int(row2) + img1.shape[0]), (0, 255, 0), 1)
        cv2.imwrite(join('outputs', 'matches', imgNames, 'match_{:04d}.jpg'.format(i+1)), imgLine)
    cv2.imwrite(join('outputs', 'matches', imgNames, 'all_matches.jpg'), all_matches)

def resize(img1_pth, img2_pth):
    img1 = cv2.imread(join('images', img1_pth))
    img2 = cv2.imread(join('images', img2_pth))
    resized1, resized2 = False, False

    while img1.shape[0] * img1.shape[1] > _MAX_SIZE:
        img1 = cv2.pyrDown(img1)
        resized1 = True

    while img2.shape[0] * img2.shape[1] > _MAX_SIZE:
        img2 = cv2.pyrDown(img2)
        resized2 = True
    
    if resized1:
        img1_pth = ''.join(img1_pth.split('.')[:-1]) + '_resized.' + img1_pth.split('.')[-1]
        cv2.imwrite(join('images', img1_pth), img1)
    if resized2:
        img2_pth = ''.join(img2_pth.split('.')[:-1]) + '_resized.' + img2_pth.split('.')[-1]
        cv2.imwrite(join('images', img2_pth), img2)
    return img1_pth, img2_pth
