import cv2 
import numpy as np 
from os.path import join
from time import time
from utils import circular_convolution
from utils import save_gaussian_scale_space
from utils import save_difference_of_gaussians
from utils import draw_keypoints

_LAMBDA_ORI = 1.5
_NUM_BINS_GRAD_ORI_HIST = 36
_THRESH_GRAD_ORI_HIST = 0.8
_NUM_HIST_NORM_PATCH = 4
_NUM_BINS_DESCR_HIST = 8
_LAMBDA_DESCR = 6

def compute_gaussian_scale_space(img, num_octaves, scales, sigma_seed, sigma_init):
    gaussians = [[] for _ in range(num_octaves)]
    # Compute seed image
    img_upsampled = cv2.resize(img, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    img_base = cv2.GaussianBlur(img_upsampled, (0, 0), 2 * (sigma_seed**2 - sigma_init**2)**0.5)
    gaussians[0].append(img_base)

    # Compute other octaves and scales
    for i in range(num_octaves):
        for j in range(1, scales+3):
            sigma = 2 * sigma_seed * (2**(2*j/scales) - 2**(2*(j-1)/scales))**0.5
            gaussian = cv2.GaussianBlur(gaussians[i][j-1], (0,0), sigma)
            gaussians[i].append(gaussian)
        if i < num_octaves-1:
            base = gaussians[i][scales][::2, ::2]
            gaussians[i+1].append(base)
    
    return gaussians
    
def compute_difference_of_gaussians(gaussians):
    num_octaves = len(gaussians)
    scales = len(gaussians[0])-3
    dogs = [[] for _ in range(num_octaves)]
    for i in range(num_octaves):
        for j in range(scales+2):
            dogs[i].append(gaussians[i][j+1] - gaussians[i][j])
    return dogs

def get_discrete_extremas(dogs):
    extremas = {}
    for o in range(len(dogs)):
        for s in range(1, len(dogs[o])-1):
            top = dogs[o][s+1]
            mid = dogs[o][s]
            bot = dogs[o][s-1]
            for r in range(1, mid.shape[0]-1):
                for c in range(1, mid.shape[1]-1):
                    # The horribly looking if statement shows better perofrmance
                    # than the other cleaner method commented out below
                    if (
                        mid.item(r, c) > top.item(r-1, c-1) and
                        mid.item(r, c) > top.item(r-1, c) and
                        mid.item(r, c) > top.item(r-1, c+1) and
                        mid.item(r, c) > top.item(r, c-1) and
                        mid.item(r, c) > top.item(r, c) and
                        mid.item(r, c) > top.item(r, c+1) and
                        mid.item(r, c) > top.item(r+1, c-1) and
                        mid.item(r, c) > top.item(r+1, c) and
                        mid.item(r, c) > top.item(r+1, c+1) and
                        mid.item(r, c) > mid.item(r-1, c-1) and
                        mid.item(r, c) > mid.item(r-1, c) and
                        mid.item(r, c) > mid.item(r-1, c+1) and
                        mid.item(r, c) > mid.item(r, c-1) and
                        mid.item(r, c) > mid.item(r, c+1) and
                        mid.item(r, c) > mid.item(r+1, c-1) and
                        mid.item(r, c) > mid.item(r+1, c) and
                        mid.item(r, c) > mid.item(r+1, c+1) and
                        mid.item(r, c) > bot.item(r-1, c-1) and
                        mid.item(r, c) > bot.item(r-1, c) and
                        mid.item(r, c) > bot.item(r-1, c+1) and
                        mid.item(r, c) > bot.item(r, c-1) and
                        mid.item(r, c) > bot.item(r, c) and
                        mid.item(r, c) > bot.item(r, c+1) and
                        mid.item(r, c) > bot.item(r+1, c-1) and
                        mid.item(r, c) > bot.item(r+1, c) and
                        mid.item(r, c) > bot.item(r+1, c+1)
                    ) or (
                        mid.item(r, c) < top.item(r-1, c-1) and
                        mid.item(r, c) < top.item(r-1, c) and
                        mid.item(r, c) < top.item(r-1, c+1) and
                        mid.item(r, c) < top.item(r, c-1) and
                        mid.item(r, c) < top.item(r, c) and
                        mid.item(r, c) < top.item(r, c+1) and
                        mid.item(r, c) < top.item(r+1, c-1) and
                        mid.item(r, c) < top.item(r+1, c) and
                        mid.item(r, c) < top.item(r+1, c+1) and
                        mid.item(r, c) < mid.item(r-1, c-1) and
                        mid.item(r, c) < mid.item(r-1, c) and
                        mid.item(r, c) < mid.item(r-1, c+1) and
                        mid.item(r, c) < mid.item(r, c-1) and
                        mid.item(r, c) < mid.item(r, c+1) and
                        mid.item(r, c) < mid.item(r+1, c-1) and
                        mid.item(r, c) < mid.item(r+1, c) and
                        mid.item(r, c) < mid.item(r+1, c+1) and
                        mid.item(r, c) < bot.item(r-1, c-1) and
                        mid.item(r, c) < bot.item(r-1, c) and
                        mid.item(r, c) < bot.item(r-1, c+1) and
                        mid.item(r, c) < bot.item(r, c-1) and
                        mid.item(r, c) < bot.item(r, c) and
                        mid.item(r, c) < bot.item(r, c+1) and
                        mid.item(r, c) < bot.item(r+1, c-1) and
                        mid.item(r, c) < bot.item(r+1, c) and
                        mid.item(r, c) < bot.item(r+1, c+1)
                    ):
                        extremas[(o, s, r, c)] = True
                    
                    # current_pixel = mid.item(r, c)
                    # top_neighborhood = top[r-1:r+2, c-1:c+2]
                    # mid_neighborhood = mid[r-1:r+2, c-1:c+2]
                    # bot_neighborhood = bot[r-1:r+2, c-1:c+2]
                    # neighborhood = np.asarray([top_neighborhood, mid_neighborhood, bot_neighborhood])
                    # if current_pixel == np.min(neighborhood) or current_pixel == np.max(neighborhood):
                    #     extremas[(o, s, r, c)] = True
    return extremas

def refine_extremas(extremas, dogs, scales, sigma_seed):
    refined_extremas = {}
    contrast_thresh = (2**(1/scales) - 1) / (2**(1/3) - 1) * 0.015

    for key in extremas:
        o, s, r, c = key
        count = 0
        height = dogs[o][s].shape[0]
        width = dogs[o][s].shape[1]

        if abs(dogs[o][s].item(r,c)) < 0.8 * contrast_thresh:
            continue

        while count < 5:
            count += 1

            grad_s = (dogs[o][s+1].item(r,c) - dogs[o][s-1].item(r,c)) / 2
            grad_r = (dogs[o][s].item(r+1,c) - dogs[o][s].item(r-1,c)) / 2
            grad_c = (dogs[o][s].item(r,c+1) - dogs[o][s].item(r,c-1)) / 2

            # Gradient vector
            gradient = np.asarray([grad_s, grad_r, grad_c])
            gradient_neg = np.asarray([-grad_s, -grad_r, -grad_c])

            grad_ss = dogs[o][s+1].item(r,c) + dogs[o][s-1].item(r,c) - 2 * dogs[o][s].item(r,c)
            grad_rr = dogs[o][s].item(r+1,c) + dogs[o][s].item(r-1,c) - 2 * dogs[o][s].item(r,c)
            grad_cc = dogs[o][s].item(r,c+1) + dogs[o][s].item(r,c-1) - 2 * dogs[o][s].item(r,c)

            grad_sr = (dogs[o][s+1].item(r+1,c) - dogs[o][s+1].item(r-1,c) - dogs[o][s-1].item(r+1,c) + dogs[o][s-1].item(r-1,c)) / 4
            grad_sc = (dogs[o][s+1].item(r,c+1) - dogs[o][s+1].item(r,c-1) - dogs[o][s-1].item(r,c+1) + dogs[o][s-1].item(r,c-1)) / 4
            grad_rc = (dogs[o][s].item(r+1,c+1) - dogs[o][s].item(r+1,c-1) - dogs[o][s].item(r-1,c+1) + dogs[o][s].item(r-1,c-1)) / 4

            hessian = np.asarray([
                [grad_ss, grad_sr, grad_sc],
                [grad_sr, grad_rr, grad_rc],
                [grad_sc, grad_rc, grad_cc]
            ])

            delta = np.dot(np.linalg.inv(hessian), gradient_neg)

            if np.max(np.abs(delta)) >= 0.6:
                s = int(round(s + delta[0]))
                r = int(round(r + delta[1]))
                c = int(round(c + delta[2]))
                if s >= 1 and s < len(dogs[o])-1 and r >= 1 and r < height-1 and c >= 1 and c < width-1:
                    continue
                else:
                    break

            sigma = 2**o * sigma_seed * 2**((delta[0]+s)/scales)
            row = 2**(o-1) * (delta[1]+r)
            col = 2**(o-1) * (delta[2]+c)

            interpolated_extremum = dogs[o][s].item(r,c) + 0.5 * np.dot(delta, gradient)
            good_contrast = abs(interpolated_extremum) > contrast_thresh
            not_on_edge = (grad_rr + grad_cc)**2 / (grad_rr * grad_cc - grad_rc**2) < 12.1
            
            if good_contrast and not_on_edge:
                refined_extremas[(o, s, r, c)] = (interpolated_extremum, sigma, row, col)

            break

    return refined_extremas

def get_reference_orientations(gaussians, keypoints):
    filtered_keypoints = {}
    for (o, s, r, c) in keypoints:
        height = gaussians[o][s].shape[0]
        width = gaussians[o][s].shape[1]

        dog, sigma, row, col = keypoints[(o, s, r, c)]
        d = 2 ** (o-1)
        patch_size = int(np.ceil(3 * _LAMBDA_ORI * sigma / 2**(o-1)))
        patch_top = r - patch_size
        patch_bot = r + patch_size + 1
        patch_left = c - patch_size
        patch_right = c + patch_size + 1

        if patch_top < 1 or patch_bot >= height or patch_left < 1 or patch_right >= width:
            continue
        
        histogram = [0 for _ in range(_NUM_BINS_GRAD_ORI_HIST)]
        for i in range(patch_top, patch_bot):
            for j in range(patch_left, patch_right):
                grad_r = (gaussians[o][s].item(i+1, j) - gaussians[o][s].item(i-1, j)) / 2
                grad_c = (gaussians[o][s].item(i, j+1) - gaussians[o][s].item(i, j-1)) / 2
                grad_magn = (grad_r**2 + grad_c**2) ** 0.5
                weight = np.exp(-((i*d-row)**2 + (j*d-col)**2) / (2 * (_LAMBDA_ORI*sigma)**2))
                weighted_grad_magn = grad_magn * weight
                
                orientation = np.floor( ((np.arctan2(grad_r, grad_c) + 2*np.pi) % (2*np.pi)) * _NUM_BINS_GRAD_ORI_HIST / (2 * np.pi))

                histogram[int(orientation)] += weighted_grad_magn

        histogram = circular_convolution(histogram, 6)
        global_max = max(histogram)
        orientations = []
        for i in range(len(histogram)):
            left = (i-1+len(histogram)) % len(histogram)
            right = (i+1) % len(histogram)

            if not (histogram[i] > histogram[left] and histogram[i] > histogram[right] and histogram[i] >= _THRESH_GRAD_ORI_HIST*global_max):
                continue
            reference_orientation = 2 * np.pi * i / len(histogram) + np.pi/len(histogram) * (histogram[left]-histogram[right]) / (histogram[left] + histogram[right] - 2*histogram[i])
            orientations.append(reference_orientation)
        filtered_keypoints[(o, s, r, c)] = (dog, sigma, row, col, orientations)
    return filtered_keypoints

def get_descriptors(gaussians, keypoints):
    descriptors = {}
    patch_size_coef = 2**0.5 * _LAMBDA_DESCR  * (_NUM_BINS_DESCR_HIST+1)/_NUM_BINS_DESCR_HIST
    norm_patch_size = _LAMBDA_DESCR * (_NUM_BINS_DESCR_HIST+1)/_NUM_BINS_DESCR_HIST
    norm_term = 1 - (1+_NUM_BINS_DESCR_HIST)/2
    near_histogram = 2 * _LAMBDA_DESCR / _NUM_BINS_DESCR_HIST

    mn_norms = [i+norm_term for i in range(_NUM_HIST_NORM_PATCH)]

    for (o, s, r, c) in keypoints:
        _, sigma, row, col, thetas = keypoints[(o, s, r, c)]
        d = 2 ** (o-1)
        height = gaussians[o][s].shape[0]
        width = gaussians[o][s].shape[1]

        patch_size = int(np.ceil(patch_size_coef * sigma / d))
        patch_top = r - patch_size
        patch_bot = r + patch_size + 1
        patch_left = c - patch_size
        patch_right = c + patch_size + 1

        if patch_top < 1 or patch_bot >= height or patch_left < 1 or patch_right >= width:
            continue
        for theta in thetas:
            histograms = [[[0 for _ in range(_NUM_BINS_DESCR_HIST)] for _ in range(_NUM_HIST_NORM_PATCH)] for _ in range(_NUM_HIST_NORM_PATCH)]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for i in range(patch_top, patch_bot):
                for j in range(patch_left, patch_right):
                    row_norm = ((i*d - row) * cos_theta + (j*d - col) * sin_theta) / sigma
                    col_norm = (-(i*d - row) * sin_theta + (j*d - col) * cos_theta) / sigma
                    if max(abs(row_norm), abs(col_norm)) < norm_patch_size:
                        grad_r = (gaussians[o][s].item(i+1, j) - gaussians[o][s].item(i-1, j)) / 2
                        grad_c = (gaussians[o][s].item(i, j+1) - gaussians[o][s].item(i, j-1)) / 2
                        theta_norm = (np.arctan2(grad_r, grad_c) - theta + 4 * np.pi) % (2*np.pi) / (2*np.pi) * _NUM_BINS_DESCR_HIST

                        grad_magn = (grad_r**2 + grad_c**2) ** 0.5
                        weight = np.exp(-((i*d-row)**2 + (j*d-col)**2) / (2 * (_LAMBDA_DESCR*sigma)**2))
                        weighted_grad_magn = grad_magn * weight

                        valid_ms = [abs(row_norm/near_histogram - mn_norms[m]) for m in range(_NUM_HIST_NORM_PATCH)]
                        valid_ns = [abs(col_norm/near_histogram - mn_norms[m]) for m in range(_NUM_HIST_NORM_PATCH)]
                        valid_ks = [(k - theta_norm + _NUM_BINS_DESCR_HIST) % _NUM_BINS_DESCR_HIST for k in range(_NUM_BINS_DESCR_HIST)]
                        for m in range(_NUM_HIST_NORM_PATCH):
                            for n in range(_NUM_HIST_NORM_PATCH):
                                for k in range(_NUM_BINS_DESCR_HIST):
                                    if valid_ms[m] <= 1 and valid_ns[n] <= 1 and valid_ks[k] < 1:
                                        histograms[m][n][k] += (1-valid_ms[m]) * (1-valid_ns[n]) * (1-valid_ks[k]) * weighted_grad_magn

            descriptor = np.asarray(histograms).flatten()
            
            euclidean = np.linalg.norm(descriptor)
            for i in range(len(descriptor)):
                descriptor[i] = min(descriptor[i], 0.2 * euclidean)
            euclidean = np.linalg.norm(descriptor)
            for i in range(len(descriptor)):
                descriptor[i] = min(np.floor(512 * descriptor[i] / euclidean), 255)
            
            descriptors[(sigma, row, col, theta)] = descriptor

    return descriptors

def sift(img_name, num_octaves=8, scales=3, sigma_seed=0.8, inter_sample_distance_init=0.5, sigma_init=0.5, save_pyramids=False):
    # Read the image.
    img_pth = join('images', img_name)
    img = cv2.imread(img_pth)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray / 255.0

    # Build scale space of images blurred with Gaussian filters.
    time1 = time()
    gaussians = compute_gaussian_scale_space(img_gray, num_octaves, scales, sigma_seed, sigma_init)
    time2 = time()
    print('Time to build scale space: {:.2f}'.format(time2-time1))
    
    # Build scale space of Laplacian Gaussian images approximated by computing
    # difference of images blurred with Gaussian fiters.
    dogs = compute_difference_of_gaussians(gaussians)
    time3 = time()
    print('Time to compute DoG: {:.2f}'.format(time3-time2))
    
    # Save scale space of Gaussians and Diff of Gaussians for demonstration.
    if save_pyramids:
        save_gaussian_scale_space(gaussians)
        save_difference_of_gaussians(dogs)
        time3 = time()

    # Locate the local extrema pixels in 3D DoG space.
    extremas = get_discrete_extremas(dogs)
    time4 = time()
    print('Time to find discrete extremas: {:.2f}'.format(time4-time3))

    # Refine the obtained extrema points by interpolating them within pixels 
    # and filtering out bad keypoints with low contrast or location.
    keypoints = refine_extremas(extremas, dogs, scales, sigma_seed)
    time5 = time()
    print('Time to refine keypoints: {:.2f}'.format(time5-time4))
    
    # For each keypoint, compute its neighborhood gradient magnitudes and orientations.
    keypoints_orientation = get_reference_orientations(gaussians, keypoints)
    time6 = time()
    print('Time to assign orientations: {:.2f}'.format(time6-time5))

    # For each keypoint, compute the 128-bit normalized descriptor vector.
    descriptors = get_descriptors(gaussians, keypoints_orientation)
    time7 = time()
    print('Time to compute descriptors: {:.2f}'.format(time7-time6))
    
    print('Time of the total SIFT: {:.2f}'.format(time7-time1))
    draw_keypoints(img_name, keypoints)
    return descriptors