# Image Feature Matcher

## Overview

This repository contains the code for extracting features from 2 different images of the same scene/object and matching them. The features are extracted by implementing the Scale Invariant Feature Transform (SIFT) algorithm. The original paper of this algorithm, that was intrduced by David G. Lowe, can be found [here](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).

## SIFT algorithm

The algorithm can be split into 5 steps:

1.  Building scale space using Gaussian filters;
2.  Computing difference of the images at different scales;
3.  Keypoint localization by scanning the images obtained from the previous step;
4.  Orientation assignment for each keypoint;
5.  Computing 128 dimensional vector descriptor for each keypoint.

The obtained descriptors are invariant to the scale, orientation, and illumination of their corresponding feature keypoints. In other words, if you have 2 images of the same scene or object that differ in scale (e.g., one photo was zoomed in), or in orientation (e.g., the object in the other image was rotated), or in illumination (e.g., one image is brighter than the other), the descriptors of the same patches in the images will be almost identical due to the mentioned invariances. 

Such kind of algorithm can be used for matching the features of multiple images, which is espeacially usefull for automatic image stitching and making nice-looking panoramic pictures.

## Using the program

### Setup

To run the code you need to have the following setup:

*   Python version 3.*
*   OpenCV version 3.*
*   Numpy (usually comes with OpenCV)

### Input images

To test the scale, orientation, and illumination invariances, you can take pictures in the following way:

*   Take a photo of some scene or object with distinct features
*   Then, take a photo again but with increased zoom and/or rotate your camera and/or add some light

After taking the pictures, save them to the `images` directory. There are already some images in the directory that you can use to test the code.

### Runnign the code

After uploading the images to the `images` directory and installing all required libraries, you can run the code as follows:

`$ python main.py`

You will be asked to enter the file names of 2 images whose features you want to match. The code then will run for about 2-3 minutes to complete. It will create an `outputs` directory, where you will see the images that show all keypoints for input images and a `matches` directory that contains all matched keypoints. If it is empty or the directory was not created, then no matches were found.

During the execution you will see how much time was spent on each step of the SIFT algorithm:

```
Welcome to the feature matcher backed by SIFT algorithm! Please enter the file names of the images whose features you want to match. Please make sure that both images are inside the `images` directory.
Image 1 (don't forget to write the file extension as well): img1.jpg 
Image 2 (don't forget to write the file extension as well): img2.jpg
Time to resize images: 0.37


Time to build scale space: 0.06
Time to compute DoG: 0.02
Time to find discrete extremas: 5.50
Time to refine keypoints: 0.05
Time to assign orientations: 2.27
Time to compute descriptors: 37.68
Time of the total SIFT: 45.72

Time to SIFT first image: 45.75


Time to build scale space: 0.05
Time to compute DoG: 0.01
Time to find discrete extremas: 5.53
Time to refine keypoints: 0.05
Time to assign orientations: 2.13
Time to compute descriptors: 35.77
Time of the total SIFT: 43.55

Time to SIFT second image: 43.57


Time to find all matches: 2.79
Time to draw all matches: 1.88
--------------------------------------------------
Total execution time: 94.35
```
