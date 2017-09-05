# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:31:05 2017

"""


import numpy as np
import os
from scipy import ndimage

def load_disp(path):
    return np.round(ndimage.imread(path,flatten=True)/256.0)
    
def load_all_images(left_paths,right_paths):
    left_images=[]
    right_images=[]

    print("Loading and preprocessing images...")
    for i in range(len(left_paths)):
        left_images = processs_input_image(ndimage.imread(left_paths[i]))
        right_images = processs_input_image(ndimage.imread(right_paths[i]))

    print("done!")
    return left_images,right_images

def processs_input_image(image):
    image=np.array(image,dtype=np.float32)
    image=(image-np.mean(image))/np.std(image)
    return image

def load_testset(im_path):
    image_left_list = []
    image_right_list = []
    image_left_list.append(os.path.join(im_path, "retif_L5252.png"))
    image_right_list.append(os.path.join(im_path, "retif_R5252.png"))
        
    left_images,right_images= load_all_images(image_left_list,image_right_list)
 
    return left_images,right_images

if __name__ == '__main__':

    Dir = os.getcwd() 
    im_path = "tensorflow_workstation/anita/porcine/allf2/"
left_images,right_images =  load_testset(im_path)
np.save('tensorflow_workstation/anita/porcine/allf2/L5252',left_images)
np.save('tensorflow_workstation/anita/porcine/allf2/R5252',right_images)
    
