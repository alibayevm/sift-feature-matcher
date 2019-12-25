from sift import sift
from os.path import isdir
from os import makedirs
from utils import resize, match_descriptors, draw_matches
from time import time

if __name__ == "__main__":
    if not isdir('outputs'):
        makedirs('outputs')
    print('Welcome to the feature matcher backed by SIFT algorithm! Please enter '+
          'the file names of the images whose features you want to match. Please '+
          'make sure that both images are inside the `images` directory.')
    img1 = input('Image 1 (don\'t forget to write the file extension as well): ')
    img2 = input('Image 2 (don\'t forget to write the file extension as well): ')
    
    time_one = time()
    img1, img2 = resize(img1, img2)
    time_two = time()
    print('Time to resize images: {:.2f}\n\n'.format(time_two-time_one))
    desc1 = sift(img1, save_pyramids=True)
    time_three = time()
    print('\nTime to SIFT first image: {:.2f}\n\n'.format(time_three-time_two))
    desc2 = sift(img2)
    time_four = time()
    print('\nTime to SIFT second image: {:.2f}\n\n'.format(time_four-time_three))
    matches = match_descriptors(desc1, desc2)
    time_five = time()
    print('Time to find all matches: {:.2f}'.format(time_five-time_four))
    
    draw_matches(matches, img1, img2)
    time_six = time()
    print('Time to draw all matches: {:.2f}'.format(time_six-time_five))
    print('-'*50)
    print('Total execution time: {:.2f}'.format(time_six-time_one))