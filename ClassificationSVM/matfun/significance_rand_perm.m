%% Test to get significance of balanced accuracy of model using random permutation
% for SVM
clear
load('SVM_results_SST4_HQ_5whales_50it','t','params','data')
data = best_mdl.data; % Time-frequency ridge data
labels = best_mdl.labels; % Labels corresponding to the whale tracks (clusters)
clust_theo2 = best_mdl.clust_theo_2; % Cluster numbers (same as labels)

for nn = 1:1000
    disp('%%%%%')
    disp(['Permutation # ' num2str(nn)])
    disp('%%%%%')

    randidx = randperm(size(data,1)); % Random shuffling of the labels
    tmp_labels = labels(randidx);
    tmp_clust = clust_theo2(randidx);

    c = cvpartition(tmp_clust,"Holdout",0.1,"Stratify",true);
    idxTrain = training(c); idxTest = test(c);
    clust_test = tmp_clust(idxTest); clust_train = tmp_clust(idxTrain);
    labels_train = tmp_labels(idxTrain);
    data_train = data(idxTrain,:);
    
    Mdl = fitcecoc(data_train,labels_train,'Learners',t,'Coding','onevsone','OptimizeHyperparameters',params,...
    'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',200,'ShowPlots',false,'Verbose',0));
    close all

    ypred = predict(Mdl,data); % Get model prediction for all samples

    for ii = 1:size(data,1)
        clust(ii,1) = str2double(ypred{ii}(2));
    end
    
    [~,bacc_train(nn)] = calc_bacc(clust_train,clust(idxTrain));
    [~,bacc_test(nn)] = calc_bacc(clust_test,clust(idxTest));
    
    CVMdl = crossval(Mdl,'KFold',4); % Cross-validate Mdl using n-fold cross-validation.
    genError(1) = kfoldLoss(CVMdl); % Estimate the generalized classification error
    CVMdl = crossval(Mdl,'KFold',5);
    genError(2) = kfoldLoss(CVMdl);
    CVMdl = crossval(Mdl,'KFold',10);
    genError(3) = kfoldLoss(CVMdl);
    genError = mean(genError);

    cverror(nn) = genError;

    clear randidx tmp_* Mdl ypred clust CVMdl genError c idxT* labels_train
    clear data_train clust_test clust_train 
end

bacc_train_m = mean(bacc_train);
bacc_test_m = mean(bacc_test);

% Balanced accuracy at p-value = 0.001 (99.9% percentile)
P = prctile(bacc_test,99.9);

save permutation_results


