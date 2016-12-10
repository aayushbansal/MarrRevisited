% --------------------------------------------------------------------------- %
% MarrRevisited - Surface Normal Estimation
% Copyright (c) 2016 Adobe Systems Incorporated and Carnegie Mellon University. 
% All rights reserved.[see LICENSE for details]
% -------------------------------------------------------------------------- %

% Written by Aayush Bansal. Please contact ab.nsit@gmail.com
% demo code to use the surface normal mode --
clc; clear all;

%
conv_cache = ['./cachedir/demo_results/'];
if(~isdir(conv_cache))
        mkdir(conv_cache);
end

% initialize caffe
NET_FILE_PATH = ['./cachedir/surface_normal_models/'];
net_file     = [NET_FILE_PATH, 'best_model.caffemodel'];
DEPLOY_FILE_PATH = ['./net/conv/'];
deploy_file  = [DEPLOY_FILE_PATH, 'deploy.prototxt']; 

% set the gpu --
gpu_id = 0;
caffe.reset_all;
caffe.set_device(gpu_id);
caffe.set_mode_gpu;
net = caffe.Net(deploy_file, net_file, 'test');

cnn_input_size = 224;
crop_height = 224; crop_width = 224;
image_mean = cat(3,  103.9390*ones(cnn_input_size),...
		     116.7700*ones(cnn_input_size),...
		     123.6800*ones(cnn_input_size));

% read the image set for NYU
img_data = {'demo/img_000001.jpg', 'demo/img_000002.jpg'};

% for each image in the img_set
for i = 1:length(img_data)

	display(['Image : ', img_data{i}]);
	ith_Img = im2uint8(imread(img_data{i}));

	%
        save_file_name = [conv_cache, strrep(img_data{i}, '.jpg', '')];
        if(exist([save_file_name, '.mat'], 'file'))
                continue;
        end
	 
        j_ims = single(ith_Img(:,:,[3 2 1]));
        j_tmp = imresize(j_ims, [cnn_input_size, cnn_input_size], ...
                           'bilinear', 'antialiasing', false);
        j_tmp = j_tmp - image_mean;
        ims(:,:,:,1) = permute(j_tmp, [2 1 3]);	

        snd(:,:,:,1) = (1/sqrt(3))*ones(cnn_input_size+200, cnn_input_size+200,3);
        depd(:,:,:,1) = zeros(cnn_input_size+200, cnn_input_size+200);

        %
        net.blobs('data0').reshape([crop_height+200, crop_width+200, 3, 1]);
        net.blobs('data2').reshape([crop_height+200, crop_width+200, 1, 1]);
        net.blobs('data1').reshape([crop_height+200, crop_width+200, 3, 1]);


        input_data = zeros(crop_height+200,crop_width+200,3,1);
        input_data(101:crop_width+100, 101:crop_width+100, :, 1) = ims;
        depd(101:crop_width+100, 101:crop_width+100, :, 1) = 1;
        net.blobs('data0').set_data(input_data);
        net.blobs('data2').set_data(depd);
        net.blobs('data1').set_data(snd);

        net.forward_prefilled();
        out = net.blobs('fc8_hcol').get_data();

        %%
        f2 = out';
        f2 = reshape(f2, [224, 224,3]);
        f2 = permute(f2, [2,1,3]);

        % normalize
        nx = f2(:,:,1); ny = f2(:,:,2); nz = f2(:,:,3);
        N = (nx.^2 + ny.^2 + nz.^2).^0.5 + eps;
        nx = nx./N; ny = ny./N; nz = nz./N;

        predns = cat(3, nx, ny, nz);
        predns = imresize(predns,...
                 [size(ith_Img,1), size(ith_Img,2)]);
        pred_N = (predns(:,:,1).^2 + predns(:,:,2).^2 +...
                                 predns(:,:,3).^2).^0.5 + eps;
        predns(:,:,1) = predns(:,:,1)./pred_N;
        predns(:,:,2) = predns(:,:,2)./pred_N;
        predns(:,:,3) = predns(:,:,3)./pred_N;

        predns_vis = uint8(255*(max(min(predns,1),-1)+1)/2);

        % dump the nx/ny/nz
        predns = single(predns);
        imwrite(predns_vis, [save_file_name, '.png']);
        save([save_file_name, '.mat'], 'predns')

end

% reset caffe
caffe.reset_all;
