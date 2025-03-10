% Apply t-SNE clustering to the time-frequency ridge data from Bryde's whale calls
clear
load('SVM_results_SST4_HQ_5whales_50it','best_mdl')
data = best_mdl.data; % Time-frequency ridge data
clust_theo2 = best_mdl.clust_theo2; % Labels

idx1 = find(clust_theo2 == 5); clust_theo2(idx1) = 4;
idx1 = find(clust_theo2 == 6); clust_theo2(idx1) = 5;

figure
clas = tsne(data,'Algorithm','exact','Distance','cityblock','NumDimensions',2,...
        'NumPCAComponents',0,'Perplexity',30,'Standardize',true);
gscatter(clas(:,1),clas(:,2),clust_theo2)
xlabel('Axis 1'); ylabel('Axis 2'); legend('boxoff')
