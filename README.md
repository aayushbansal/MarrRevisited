# Marr Revisited: 2D-3D Alignment via Surface Normal Prediction
Created by Aayush Bansal, Bryan Russell, and Abhinav Gupta at CMU and Adobe Research.

## Introduction 

This work is a part of our CVPR-2016 paper on 2D-3D model alignment via surface normal prediction. In this work, we predict detailed surface normals through the non-linear optimization of sparse hypercolumn features. The idea of sparsely sampling hypercolumn features allowed the non-linear optimization by a multi-layer perceptron. This work is the current state-of-the-art (as of July 2016) on NYUD dataset for both global and local layout. For more details, refer to the text in our paper http://www.cs.cmu.edu/~aayushb/marrRevisited/. Here, we give step-by-step instruction to use our code for training and predicting surface normal maps.

## How to use this code?

1. Installing Caffe Toolbox - We have included the required source codes with the original caffe codes. To use our stuff, one needs to install caffe in the given folder. 

2. Once the caffe is installed, we have provided the prototxt files for training or predicting the surface normal on a given 2D image in the experiments/net folder. There are two folders - a. conv; b. deconv. In our paper, we used the ones in 'a. conv'. The 'b. deconv' folder is provided if there are interest in using the code via a deconvolution layer. The required details are provided in the respective folder. Note that we resized input image to 224x224, and used only single scale for both training and evaluation.

3. The required caffe models (VGG-16 model converted to a fully convolutional model, and our best model for surface normal estimation) can be downloaded from <http://www.cs.cmu.edu/~aayushb/marrRevisited/surface_normal_models.tar.gz>. Once you have these models, you are good to use our code for both deployment and training your own model (TYOM). Feel free to contact authors if you have difficulty TYOM.

## Using it with NYU-v2 depth dataset - 
In this work, we showed evaluation on NYU-v2 depth dataset. Following we describe how to use our dataset with this dataset: 

### Training a model -
1. Download the file <http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat> for 1449 labeled examples, and <http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip> for raw video frames. Please cite their work if you use this dataset. The train, val, and test set splits are available here at <http://www.cs.cmu.edu/~aayushb/marrRevisited/data/NYU_splits.tar.gz>.

2. Surface Normal from Kinect - We used the surface normal provided by Ladicky et al. (ECCV 2014) for 1449 images. Those normal maps are available here at <https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip>. For the raw video frames, we used the code from Wang et al. (CVPR 2015) as it was computationally too expensive to compute the normals using the approach of Ladicky et al. on all video frames. Note that in our work, we ignored the pixels belonging to invalid depth data. Therefore, other than the normal maps and images, one requires a map of valid pixels for each image.

3. Once you have corresponding training images, normal maps, and valid depth data information (provided that you have successfully installed the code and downloaded the models) --  modify the paths mentioned in scripts run.sh and train.prototxt in the folder <normals/net/conv> to refer to data location. After these changes, simply running run.sh  to train our model.

Point to Note - As mentioned in our paper, we resized the images to 224x224. Similarly we resized the normals to 224x224 and then normalized them. The valid pixel depth map is also resized to same value. To provide zero padding, we padded the images by 100 pixels. 

### Using our trained model 
Once you have successfully installed the code and downloaded the models, it can be easily used to compute normal maps using the deploy file.

## Evaluation Script
The evaluation script is in folder <normals/eval>. The path for data might need a change in the scripts. Our results are available at <ttp://www.cs.cmu.edu/~aayushb/marrRevisited/data/nyu_test.zip>. 

## Contact 
Please contact Aayush Bansal in case of any queries or comments about the code.

## LICENSE
This code is copyrighted by the authors, CMU, and Adobe, and is for non-commercial research purposes only. Please contact the authors, CMU, or Adobe if you are interested in licensing for commercial purposes.

## CITATION 
Please cite our paper if you use this code for any publishable academic research:

@inproceedings{Bansal16,
  title={Marr {R}evisited: 2{D}-3{D} Model Alignment via Surface Normal Prediction},
  author={Bansal, Aayush and Russell, Bryan and Gupta, Abhinav},
  booktitle={CVPR},
  year={2016}
}


