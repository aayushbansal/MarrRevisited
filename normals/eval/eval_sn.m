% --------------------------------------------------------------------------- %
% MarrRevisited - Surface Normal Estimation
% Copyright (c) 2016 Adobe Systems Incorporated and Carnegie Mellon University. 
% All rights reserved.[see LICENSE for details]
% -------------------------------------------------------------------------- %

% Written by Aayush Bansal. Please contact ab.nsit@gmail.com

% use the network to predict the values of nx, ny, and nz.
% function comp_nxyz(PREDN, NET_NAME, MODEL_ITER)
clc; clear all;

% setup the task --
DUMP_CACHE = './cachedir/best_model/';

% load the nyu-dataset --
img_set = 'test';
imgLabs = load(['./dataset/NYU/img_set/',img_set,'.mat'], 'img_set');
imgLabs = imgLabs.img_set;

% evaluate the predicted sn --
[nums_e] = eval_pred_sn(DUMP_CACHE, imgLabs);
