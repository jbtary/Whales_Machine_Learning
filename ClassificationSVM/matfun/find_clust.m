% Find which cluster corresponds to which with k-means classification
% 
% IN
%   clust: cluster numbers per training example output from k-means
%   clust_theo: theoretical clusters corresponding to the training ex.
% 
% OUT
%   clust_km: best cluster combination as estimated by the percentage of
%   good matches between clust and clust_theo, saved in sv_preds
%   clust_key: cluster numbers corresponding to those in clust_theo
% 

function [clust_km,sv_preds,clust_key] = find_clust(clust,clust_theo)

clust_ind = unique(clust_theo); % Get all cluster indexes
nc = length(clust_ind); % Number of clusters

% Get all possible permutations of the clusters
clust_perm = perms(clust_ind);

sv_preds = 0; % Best preds number to update
for ii = 1:size(clust_perm,1)

    % Change cluster indexes corresponding to current combination
    clust_tmp = clust+0.1;
    for jj = 1:nc
        clust_tmp(clust_tmp==jj+0.1) = clust_perm(ii,jj);
    end
    
    % Calculate percentage of correct clustering
    good_preds = sum(clust_tmp == clust_theo)/length(clust_tmp)*100;
    
    % If the current combination works better, save it
    if sv_preds < good_preds
        sv_preds = good_preds;
        clust_km = clust_tmp;
        clust_key = clust_perm(ii,:);
    end

    clear clust_tmp good_preds
end
