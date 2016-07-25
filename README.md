# Surface Normal Estimation
Created by Aayush Bansal, Bryan Russell, and Abhinav Gupta at CMU and Adobe Research.

# Introduction 

This work is a part of our CVPR-2016 paper on 2D-3D model alignment via surface normal prediction. In this work, we predict detailed surface normals through the non-linear optimization of sparse hypercolumn features. The idea of sparsely sampling hypercolumn features allowed the non-linear optimization by a multi-layer perceptron. This work is the current state-of-the-art (as of July 2016) on NYUD dataset for both global and local layout. For more details, refer to the text in our paper http://www.cs.cmu.edu/~aayushb/marrRevisited/. 

# How to use this code?

1. Installing Caffe Toolbox - We have included the required source codes with the original caffe codes. To use our stuff, one needs to install caffe in the given folder. 

2. Once the caffe is installed, we have provided the prototxt files for training or predicting the surface normal on a given 2D image in the experiments/net folder. There are two folders - a. conv; b. deconv. In our paper, we used the ones in 'a. conv'. The 'b. deconv' folder is provided if there are interest in using the code via a deconvolution layer. The required details are provided in the respective folder. Note that we resized input image to 224x224, and used only single scale for both training and evaluation.

3. The required caffe models (VGG-16 model converted to a fully convolutional model, and our best model for surface normal estimation) can be downloaded from <http://www.cs.cmu.edu/~aayushb/marrRevisited/surface_normal_models.tar.gz>. Once you have these models, you are good to use our code for both deployment and training your own model (TYOM). Feel free to contact authors if you have difficulty TYOM.

# LICENSE
This code is copyrighted by the authors, CMU, and Adobe, and is for non-commercial research purposes only. Please contact the authors, CMU, or Adobe if you are interested in licensing for commercial purposes.

# CITATION 
Please cite our paper if you use this code for any publishable academic research:

@inproceedings{Bansal16,
  title={Marr {R}evisited: 2{D}-3{D} Model Alignment via Surface Normal Prediction},
  author={Bansal, Aayush and Russell, Bryan and Gupta, Abhinav},
  booktitle={CVPR},
  year={2016},
}


