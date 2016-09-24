% --------------------------------------------------------------------------- %
% MarrRevisited - Surface Normal Estimation
% Copyright (c) 2016 Adobe Systems Incorporated and Carnegie Mellon University. 
% All rights reserved.[see LICENSE for details]
% -------------------------------------------------------------------------- %

% Written by Aayush Bansal. Please contact ab.nsit@gmail.com
function[nums_e] = eval_pred_sn(cache_dir, cache_list)

	num_images = length(cache_list);
	for i = 1:num_images
	
		% load the file from cache
		display(['Loading image: ', num2str(i, '%06d'),'/',...
					    num2str(num_images, '%06d')]);

		% CHANGE THE NAME OF THE DATA FILE -- if 
		pred = load([cache_dir, '/', num2str(cache_list(i), '%06d'),...
							 '.mat'],  'predns');
	        gtd = load(['./dataset/NYU/GT_Normals/test/nm_',...
					 num2str(cache_list(i),'%06d') '.mat']);

		%
		NG = cat(3,gtd.nx,gtd.ny,gtd.nz);
		NV = gtd.depthValid;
		%
		NP = pred.predns;
		%normalize both to be sure
	        NG = bsxfun(@rdivide,NG,sum(NG.^2,3).^0.5);
                NP = bsxfun(@rdivide,NP,sum(NP.^2,3).^0.5);
		%compute the dot product, and keep on the valid
		DP = sum(NG.*NP,3);
		T = min(1,max(-1,DP));
		pixels{i} = T(find(NV));
	end

	E = acosd(cat(1,pixels{:}));
	nums_e = [mean(E(:)),median(E(:)),mean(E.^2).^0.5,mean(E < 11.25)*100,mean(E < 22.5)*100,mean(E < 30)*100]
	display('---------------------------------------');
	display(['Mean: ', num2str(mean(E(:)))]);
	display(['Median: ', num2str(median(E(:)))]);
	display(['RMSE: ', num2str(mean(E.^2).^0.5)]);
	display(['11.25: ', num2str(mean(E < 11.25)*100)]);
	display(['22.5: ', num2str(mean(E < 22.5)*100)]);
	display(['30: ', num2str(mean(E < 30)*100)]);
	display(['45: ', num2str(mean(E < 45)*100)]);
	display('---------------------------------------');
end
