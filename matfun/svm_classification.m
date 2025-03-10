% Using the time-frequency ridges to train machine learning models
clear
% Load the time-frequency ridges: ridges
% Load the location of the corresponding calls: locs (not used for classification)
% Load the labels of each time-frequency ridge: clust_theo_c
load('SVM_results_SST4_HQ_5whales_50it','ridges','locs','clust_theo_c')

%% SVM classification of time-frequency ridges
% Repeat the process 50 times
clearvars -except locs ridges clust_theo_c

% Choose which whales to use
[clust_theo2,ridges2,locs2] = ch_whales(clust_theo_c,ridges,locs,[1 2 3 5 6]);

% Configuration of SVM parameters
t = templateSVM('KernelFunction','gaussian','Standardize',true);

for ii = 1:50
    tic
    disp(['%%% iteration # ' num2str(ii) ' %%%'])
    % Random shuffling of the dataset
    randidx = randperm(size(ridges2,1));
    ridges3 = ridges2(randidx,:);
    clust_theo3 = clust_theo2(randidx);
    locs3 = locs2(randidx,:);
    % Generate the corresponding labels
    for jj = 1:length(clust_theo3); labels{jj,1} = ['W' num2str(clust_theo3(jj))]; end

    data = feat_scal(ridges3,1); % Standardize data: mandatory

    c = cvpartition(clust_theo3,"Holdout",0.1,"Stratify",true);
    idxTrain = training(c); idxTest = test(c);
    clust_test = clust_theo3(idxTest); clust_train = clust_theo3(idxTrain);

    labels_train = labels(idxTrain);
    data_train = data(idxTrain,:);
    
    params = hyperparameters('fitcecoc',data_train,labels_train,'svm');
    params(2,1).Optimize = true; params(2,1).Range = [1e-4 10000]; % C (boxconstraint)
    params(3,1).Optimize = true; params(3,1).Range = [1e-4 10000]; % kernel scale (gamma for Gaussian)
    params(5,1).Optimize = false; params(5,1).Range = [2 5]; % polynomial order
    params(6,1).Optimize = false; % standardize
    % Run the SVM parameter optimization for classification
    Mdl = fitcecoc(data_train,labels_train,'Learners',t,'Coding','onevsone','OptimizeHyperparameters',params,...
        'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',200,'Verbose',0)); % ,'Coding','onevsone'

    CVMdl = crossval(Mdl,'KFold',4); % Cross-validate Mdl using n-fold cross-validation.
    genError(1) = kfoldLoss(CVMdl); % Estimate the generalized classification error
    CVMdl = crossval(Mdl,'KFold',5);
    genError(2) = kfoldLoss(CVMdl);
    CVMdl = crossval(Mdl,'KFold',10);
    genError(3) = kfoldLoss(CVMdl);
    genError = mean(genError);
    cverror(ii) = genError;

    ypred = predict(Mdl,data); % Get model prediction for all samples

    for jj = 1:size(ridges3,1)
        clust(jj,1) = str2double(ypred{jj}(2));
    end

    % Balanced accuracy - train part
    [~,bacc_train(ii)] = calc_bacc(clust_train,clust(idxTrain));

    % Balanced accuracy - test part
    [~,bacc_test(ii)] = calc_bacc(clust_test,clust(idxTest));
    
    % Save the best model
    if bacc_test(ii) == max(bacc_test)
        [labelpreds,scorepreds] = kfoldPredict(CVMdl);
        figure; cm = confusionchart(labels_train,labelpreds);
        
        best_mdl.Mdl = Mdl;
        best_mdl.data = data; best_mdl.ridges2 = ridges3;
        best_mdl.clust_theo2 = clust_theo3; best_mdl.labels = labels;
        best_mdl.idxTrain = idxTrain; best_mdl.idxTest = idxTest;
        best_mdl.CVMdl = CVMdl;
        best_mdl.ypred = ypred; best_mdl.clust = clust;
        best_mdl.cm = cm;
        best_mdl.locs2 = locs3;
    end
    
    close all
    clear randidx ridges3 clust_theo3 locs3 jj labels data c idxTrain idxTest
    clear clust_test clust_train labels_train data_train Mdl CVMdl genError
    clear ypred clust labelpreds scorepreds cm
    toc
end
