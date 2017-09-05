This repository accompanies my Master's Thesis:
Learning Similarity for 3D Reconstruction of Intraoperative Environments with Convolutional Neural Networks

I trained an CNN to predict disaprity in stereo images by learning similarity. 

It includes:
- 6 trained models
- 5 .py files containing code for preprocessing, training, testing (on the Kitti data set) and predicting. 
- 3 images to viualize the architecture. 

The training/test set can be downloaded here:
http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

stereoMatchingTraining:

We trained our models on the KITTI dataset. Download the training set from the website and run load_kitti.py to convert images to .npy. Change directory in stereoMatchingTraining.py to the correct dataset and run.  

stereoMatchingTesting:

Change directory to correct validation set.

stereoMatchingPredict:

First run load_image.py to convert (any) image to .npy-format. In stereoMatchingPredict.py choose model and change directory to load the correct weights. Also change the network that should be used in make_graph(). Change directory to images that are to be predicted. 

Our best model is 7_layer_tv.

