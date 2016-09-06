# Marr Revisited: 2D-3D Alignment via Surface Normal Prediction
Created by Aayush Bansal, Bryan Russell, and Abhinav Gupta at CMU and Adobe Research.

## Introduction 

This work is a part of our CVPR-2016 paper on 2D-3D model alignment via surface normal prediction. In this work, we predict detailed surface normals through the non-linear optimization of sparse hypercolumn features. The idea of sparsely sampling hypercolumn features allowed the non-linear optimization by a multi-layer perceptron. This work is the current state-of-the-art (as of July 2016) on NYUD dataset for both global and local layout. For more details, refer to the text in our paper http://www.cs.cmu.edu/~aayushb/marrRevisited/. Here, we give step-by-step instruction to use our code for training models and estimating surface normal maps. We will release the codes for pose estimation soon.

## How to use this code?

1. Installing Caffe Toolbox - We have included the required source codes with the original caffe codes. To use our stuff, one needs to install caffe in the given folder. Please follow the instructions provided at
   ```make
      http://caffe.berkeleyvision.org/installation.html
   ```

2. The details corresponding to surface normal estimation is present in the folder "normals".
   ```make
	cd normals
   ```
   
   After successfully installing caffe, make a soft-link of caffe in toolbox folder:
   ```make
	cd toolbox/
	ln -s ../../caffe ./
	cd ..
   ```
  
   The prototxt files for training the model and predicting the surface normal map from a single 2D image are available in "net" folder:
   ```make
	 ls net
   ```
   There are two folders - a. conv; b. deconv. In our paper, we used the ones in 'a. conv'. The 'b. deconv' folder is provided if there are interest in using the code via a deconvolution layer.  

   The evaluation script is available in "eval" folder:
   ```make
	ls eval
   ```

3. The required caffe models (VGG-16 model converted to a fully convolutional model, and our best model for surface normal estimation) can be downloaded from :
   ```make
   # download the models for estimating surface normal maps or to train your own model.
   cd cachedir
   wget http://www.cs.cmu.edu/~aayushb/marrRevisited/surface_normal_models.tar.gz
   tar -xvzf surface_normal_models.tar.gz
   cd ..
   ```

4. Demo code to generate surface normal maps is available in "demo"

Specific instructions are given below to use our code for estimating surface normal maps from our trained model, and how to train a new model using NYU-v2 depth dataset.

## DEMO 

Assuming you are in the "normals/" folder, we now give instruction of how to use our code for generating surface normal maps for two random images from NYU depth dataset.
```make
   matlab
   demo_code;
```
The results can be seen here :
```make
   ls ./cachedir/demo_results/
```

## Other than Demo

In this work, we showed evaluation on NYU-v2 depth dataset. We now describe how to use our code with this dataset: 

### Training a model -
1. Download the NYU-v2 dataset: 

  ```make
  # 1449 labeled examples
  mkdir dataset
  cd dataset
  wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
  cd ..
  ```

  ```make
  # Raw video frames
  cd dataset
  wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip
  cd ..
  ```

  The train, val, and test set splits can be downloaded using: 

  ```make
  # train-val-test splits
  cd dataset
  wget http://www.cs.cmu.edu/~aayushb/marrRevisited/data/NYU_splits.tar.gz
  tar -xvzf NYU_splits.tar.gz
  cd ..
  ```

  Note that Silberman et al. (ECCV 2012) contained the train-test split. Gupta et al. (ECCV 2014) provided an extra train-val split from training split. We use the same splits (also provided above).

2. Surface Normal from Kinect - We used the surface normal provided by Ladicky et al. (ECCV 2014) for 1449 images. Those normal maps can be downloaded using: 

   ```make
   # Surface normal maps from kinect data for 1449 images.
   cd dataset
   wget https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip
   unzip nyu_normals_gt.zip
   cd ..
   ```

3. We used the code from Wang et al. (CVPR 2015) for raw video frames as it was computationally too expensive to compute the normals using the approach of Ladicky et al. on all video frames. The codes can be downloaded using
   ```make
   # Code to extract surface normal maps from kinect using Wang et al. (CVPR 2015)
   # Details to use this code are given in the folder
   cd toolbox/
   wget http://www.cs.cmu.edu/~aayushb/marrRevisited/data/kinect_normals_code.tar.gz
   tar -xvf kinect_normals_code.tar.gz
   cd ..
   # For more details about this code, see the script getNormals.m
   # All the required data is present in this folder
   ```

   Note that in our work, we ignored the pixels belonging to invalid depth data. Therefore, other than the normal maps and images, one requires a map of valid pixels for each image.

4. Data Preparation - Using the dataset, one can prepare the required data to train a new model for surface normal estimation. As mentioned in our paper, we resized the images to 224x224. Similarly we resized the normals to 224x224 and then normalized them. The valid pixel depth map is also resized to same value. To provide zero padding, we padded the images by 100 pixels.

5. Once you have corresponding training images, normal maps, and valid depth data information (provided that you have successfully installed the code and downloaded the models) --  modify the paths mentioned in scripts run.sh and train.prototxt in the folder "normals/net/conv" to refer to data location. After these changes, simply use run.sh  to train our model.
   ```make
   # Training a new model after making required modifications
   cd net/conv
   ./run.sh
   # We ran the network for 60K iterations. This is roughly 1 epoch of training data.
   ```

### Using the trained model 
Once you have successfully trained your model, it can be easily used to compute normal maps using the deploy file. See the demo code in demo/.

## Evaluation Script
The evaluation script is in folder "normals/eval". The path for data might need a change in the scripts. Our results can be downloaded from: 

  ```make
  # The surface normal maps generated using our model.
  wget http://www.cs.cmu.edu/~aayushb/marrRevisited/data/nyu_test.zip
  ```

## Contact 
Please contact Aayush Bansal in case of any queries or comments about the code.

## LICENSE
This code is copyrighted by the authors, CMU, and Adobe, and is for non-commercial research purposes only. Please contact the authors, CMU, or Adobe if you are interested in licensing for commercial purposes.

## CITATION 
Please cite our paper if you use this code for any publishable academic research:

```make
@inproceedings{Bansal16,
  title={Marr {R}evisited: 2{D}-3{D} Model Alignment via Surface Normal Prediction},
  author={Bansal, Aayush and Russell, Bryan and Gupta, Abhinav},
  booktitle={CVPR},
  year={2016}
}
```

