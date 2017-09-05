Learning Similarity for 3D Reconstruction of Intraoperative Environments with Convolutional Neural Networks

This repository includes:
- 6 trained models
- 5 .py files containing code for preprocessing, training, testing (on the Kitti data set) and predicting. 

We do not provide the test set, it can be downloaded here:
http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

stereoMatchingTraining:
We trained our models on the KITTI dataset. To run the code, run load_kitti.py first and convert images to .npy. Change directory in stereoMatchingTraining.py to the correct dataset. 

stereoMatchingTesting:
Change directory to correct validation set.

stereoMatchingPredict:
First run load_image.py to convert image to .npy-format. In stereoMatchingPredict.py choose model and change directory to load the correct weights. Also change the network that should be used in make_graph(). Change directory to images that are to be predicted. 

