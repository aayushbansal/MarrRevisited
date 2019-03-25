# Marr Revisited: 2D-3D Alignment via Surface Normal Prediction
Created by [Aayush Bansal](http://www.cs.cmu.edu/~aayushb), [Bryan Russell](http://www.bryanrussell.org/), and [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg) at CMU and Adobe Research.

Acknowledgements: a huge thanks to [Xinlei Chen](http://www.cs.cmu.edu/~xinleic) for all discussions and help.

## Introduction 

This work is a part of our CVPR-2016 paper on 2D-3D model alignment via surface normal prediction. In this work, we predict detailed surface normals through the non-linear optimization of sparse hypercolumn features. The idea of sparsely sampling hypercolumn features allowed the non-linear optimization by a multi-layer perceptron. This work is the current state-of-the-art (as of July 2016) on NYUD dataset for both global and local layout. For more details, refer to the text in our [paper](http://www.cs.cmu.edu/~aayushb/marrRevisited/). 

The codes can be downloaded using the following command:
```make 
git clone --recursive https://github.com/aayushbansal/MarrRevisited.git
cd MarrRevisited
```

Here, we give step-by-step instruction to use our code for training models and estimating surface normal maps. We will release the codes for pose estimation soon, and optimized version of surface normals sometime soon.

## Installing the Code

1. Installing Caffe Toolbox - We have included a pointer to Caffe as a submodule. Our required layers are available within this submodule. You will need to pull it using the following command (in case you want to use it for some other project):
   ```make
      git clone https://github.com/aayushbansal/caffe.git
   ```

    To install Caffe, please follow the instructions on their [project page](http://caffe.berkeleyvision.org/).

2. The details corresponding to surface normal estimation is present in the "normals/" directory.
   ```make
	cd normals
   ```
  
   The prototxt files for training the model and predicting the surface normal map from a single 2D image are available in "normals/net" directory:
   ```make
	 ls net
   ```
   There are two folders - a. conv; b. deconv. In our paper, we used the ones in 'a. conv'.


   The required caffe models (VGG-16 model converted to a fully convolutional model, and our best model for surface normal estimation) can be downloaded from :
   ```make
   # download the models for estimating surface normal maps or to train your own model.
   cd cachedir
   wget http://www.cs.cmu.edu/~aayushb/marrRevisited/surface_normal_models.tar.gz
   tar -xvzf surface_normal_models.tar.gz
   cd ..
   ```

Specific instructions are given below to use our code for estimating surface normal maps from our trained model, and how to train a new model using NYU-v2 depth dataset.

## DEMO

You can run the demo code inside MATLAB (assuming you are in the "normals/" directory):

```make
   demo_code;
```
The results can be seen here :
```make
   ls ./cachedir/demo_results/
```

## Evaluation Script
The evaluation script is in "normals/eval" directory. Our results can be downloaded using following (given you are in "normals/" directory):

  ```make
  # The surface normal maps generated using our model.
  cd cachedir/
  wget http://www.cs.cmu.edu/~aayushb/marrRevisited/data/nyu_test.zip
  unzip nyu_test.zip
  mv results best_model  
  cd ..
  ```
 
  The other thing needed for evaluation are the surface normal maps from Kinect data on test set. Please see the section below to download the data. Once you have the required data, run following inside MATLAB:
  ```make
    eval_sn
  ```


## Training your own model for surface normal prediction on NYUv2

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

  Note that [Silberman et al. (ECCV 2012)](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) contained the train-test split. [Gupta et al. (ECCV 2014)](http://people.eecs.berkeley.edu/~sgupta/pdf/rcnn-depth.pdf) provided an extra train-val split from training split. We use the same splits (also provided above).

2. Surface Normal from Kinect - We used the surface normal provided by [Ladicky et al. (ECCV 2014)](https://www.inf.ethz.ch/personal/ladickyl/normals_eccv14.pdf) for 1449 images. Those normal maps can be downloaded using: 

   ```make
   # Surface normal maps from kinect data for 1449 images.
   cd dataset
   # Original source (Unreliable download connection as of March 2019).
   # wget https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip
   # unzip nyu_normals_gt.zip

   # A clone with reliable download (as of March 2019).
   wget https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/Ladicky.tar.gz
   tar xvzf Ladicky.tar.gz
   cd ..
   ```

   The evaluation script however requires the groud truth surface normals in .mat format. These can be downloaded.
   ```bash
   wget https://www.dropbox.com/s/s85qfjq6xq9n63x/marr-revisited.zip
   unzip marr-revisited.zip
   ```

   The resulting .mat files need to be moved to `normals/dataset/NYU/GT_Normals/test`.

3. We used the code from [Wang et al. (CVPR 2015)](http://www.cs.cmu.edu/~xiaolonw/deep3d.html) for raw video frames as it was computationally too expensive to compute the normals using the approach of [Ladicky et al.](https://www.inf.ethz.ch/personal/ladickyl/normals_eccv14.pdf) on all video frames. The codes can be downloaded using
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

## Contact 
Please contact [Aayush Bansal](http://www.cs.cmu.edu/~aayushb) in case of any query or comment about the code.

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

## RELATED WORK
More details about our network architecture are available in [PixelNet](http://www.cs.cmu.edu/~aayushb/pixelNet/) paper:

```make
@article{BansalChen16,
  title={PixelNet: {T}owards a {G}eneral {P}ixel-level {A}rchitecture},
  author={Bansal, Aayush and Chen, Xinlei, and  Russell, Bryan and Gupta, Abhinav and Ramanan, Deva},
  Journal={arXiv preprint arXiv:1609.06694},
  year={2016}
}
```
