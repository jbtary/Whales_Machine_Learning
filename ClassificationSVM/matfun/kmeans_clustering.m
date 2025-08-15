% Using the time-frequency ridges to train machine learning models
clear
% Load the time-frequency ridges: ridges
% Load the location of the corresponding calls: locs (not used for classification)
% Load the labels of each time-frequency ridge: clust_theo_c
load('SVM_results_SST4_HQ_5whales_50it','ridges','locs','clust_theo_c')

%% K-means clustering of time-frequency ridges
clearvars -except locs ridges clust_theo_c

% Choose which whales to use
[clust_theo2,ridges2,locs2] = ch_whales(clust_theo_c,ridges,locs,[1 2 3 5 6]);

c = 5; % Number of clusters

% Data standardization
data = feat_scal(ridges2,1);

% Distances: L2 (sqeuclidean), L1 (cityblock) <= works better
[clust,C,sumdist] = kmeans(data,c,'Distance','cityblock','Display','final',...
    'Replicates',5);
figure; [silh,h] = silhouette(data,clust,'cityblock');
xlabel('Silhouette Value'); ylabel('Cluster')

[clust_km,acc,clust_key] = find_clust(clust,clust_theo2);

for ii = 1:length(clust_km)
    labels{ii,1} = ['W' num2str(clust_theo2(ii))];
    labelpreds{ii,1} = ['W' num2str(clust_km(ii))];
end

figure; cm = confusionchart(labels,labelpreds);

% Add balanced accuracy measure
[~,bacc_km] = calc_bacc(clust_theo2,clust_km);
