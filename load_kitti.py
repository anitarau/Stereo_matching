# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:31:05 2017

"""

import numpy as np
import os
from scipy import ndimage

def load_disp(path):
    return np.round(ndimage.imread(path,flatten=True)/256.0)
    
def load_all_images(left_paths,right_paths,disp_paths):
    left_images=[]
    right_images=[]
    disp_images=[]
    print("Loading and preprocessing images...")
    for i in range(len(left_paths)):
        left_images.append(processs_input_image(ndimage.imread(left_paths[i])))
        right_images.append(processs_input_image(ndimage.imread(right_paths[i])))
        disp_images.append(load_disp(disp_paths[i]))
    print("done!")
    return left_images,right_images,disp_images


def processs_input_image(image):
    image=np.array(image,dtype=np.float32)
    image=(image-np.mean(image))/np.std(image)
    return image

def load_kitti_2015(kitti_2015_path,nvalidation=40,shuffle=True):
    #validation_index=np.random.randint(low=0, high=199, size=nvalidation)
    deck = list(range(0, 200))
    np.random.shuffle(deck)
    validation_index=deck[0:nvalidation]
    training_list_left=[]
    training_list_right=[]
    training_list_noc_label=[]
    validation_list_left=[]
    validation_list_right=[]
    validation_list_noc_label=[]
    for i in range(0,200):
        if i in validation_index:
            validation_list_left.append(os.path.join(kitti_2015_path, "training/image_2",str(i).zfill(6)+"_10.png"))
            validation_list_right.append(os.path.join(kitti_2015_path, "training/image_3",str(i).zfill(6)+"_10.png"))
            validation_list_noc_label.append(os.path.join(kitti_2015_path, "training/disp_noc_0",str(i).zfill(6)+"_10.png"))
 
        else:
            training_list_left.append(os.path.join(kitti_2015_path, "training/image_2",str(i).zfill(6)+"_10.png"))
            training_list_right.append(os.path.join(kitti_2015_path, "training/image_3",str(i).zfill(6)+"_10.png"))
            training_list_noc_label.append(os.path.join(kitti_2015_path, "training/disp_noc_0",str(i).zfill(6)+"_10.png"))

    left_images,right_images,disp_images = load_all_images(training_list_left,training_list_right,training_list_noc_label)
    left_images_validation,right_images_validation,disp_images_validation = load_all_images(validation_list_left,validation_list_right,validation_list_noc_label)
       

    
    np.save('left_images_1',left_images)
    np.save('right_images_1',right_images)
    np.save('disp_images_1',disp_images)
    np.save('left_images_validation_1',left_images_validation)
    np.save('right_images_validation_1',right_images_validation)
    np.save('disp_images_validation_1',disp_images_validation)
    return left_images,right_images,disp_images,left_images_validation,right_images_validation,disp_images_validation

if __name__ == '__main__':
    np.random.seed(seed=123)
    Dir = os.getcwd() 
    kitti_2015_path = "datasets/kitti/data_scene_flow_2015/"
    left_images,right_images,disp_images,left_images_validation,right_images_validation,disp_images_validation = load_kitti_2015(kitti_2015_path)

    
