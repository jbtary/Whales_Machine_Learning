% Choose which whale calls to use for classification
% Maximum of 6 whales
% Number of whale calls for whale tracks: W1 71 W2 133 W3 199 W4 89 W5 89 W6
% 38
% 
% IN:
%   clust_theo: theoretical numbers of the cluster per each ridge (Nx1)
%   ridges: frequency vector for each sample (NxFreqs)
%   idx: numbers of the clusters to keep (they need to be in clust_theo)
% 
% OUT:
%   new_clust: clusters selected
%   new_rg: ridges of the clusters selected
% 

function [new_clust,new_rg,new_locs] = ch_whales(clust_theo,ridges,locs,idx)

new_clust = [];
new_rg = [];
new_locs = [];
for ii = idx
    iv = find(clust_theo == ii);
    new_clust = [new_clust;clust_theo(iv)];
    new_rg = [new_rg;ridges(iv,:)];
    new_locs = [new_locs;locs(iv,:)];
    clear iv
end
