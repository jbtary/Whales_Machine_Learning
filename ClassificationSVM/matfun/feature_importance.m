%% Two ways of checking for the influence of features on the model classification: 
% permutation importance and SHAP values

%% Calculate the permutation importance
clear
load('SVM_results_SST4_HQ_5whales_50it.mat','best_mdl')
data = best_mdl.data; % Time-frequency ridge data
Mdl = best_mdl.Mdl; % SVM model

% meas = data(best_mdl.idxTest,:); % Test set
% labels = labels(best_mdl.idxTest);

meas = data(best_mdl.idxTrain,:); % Train set
labels = best_mdl.labels(best_mdl.idxTrain);

% Get feature importance using the permutation importance (not possible to
% get directly the coefficients using other than linear SVM kernel)

% Use the test set to obtain the feature importance
[Importance,ImportancePerPermutation,ImportancePerClass] = ...
    permutationImportance(Mdl,meas,labels,...
    NumPermutations=50);

figure; set(gcf,'Position', [300 400 800 500])
subplot(2,1,1)
plot([0 3],[0 0],'--','LineWidth',2,'Color',[.8 .8 .8]); hold on
errorbar(0.03:0.03:3,Importance.ImportanceMean,...
    Importance.ImportanceStandardDeviation,'ko','LineWidth',1,'MarkerFaceColor',[0.9 0.9 0.9])
ylabel("Mean Importance")

subplot(2,1,2)
bar(0.03:0.03:3,ImportancePerClass.ImportanceMean{:,:},"stacked")
legend('W1','W2','W3','W4','W5')
ylabel("Mean Importance"); xlabel('Features (frequencies at time indexes)')

%% Calculate SHAP feature importance
clear
load('SVM_results_SST4_HQ_5whales_50it.mat','best_mdl')
data = best_mdl.data; % Time-frequency ridge data
Mdl = best_mdl.Mdl; % SVM model

% meas = data(best_mdl.idxTest,:); % Test set
% labels = labels(best_mdl.idxTest);

meas = data(best_mdl.idxTrain,:); % Train set
labels = best_mdl.labels(best_mdl.idxTrain);

% Takes about 5-10 min with parallel computing
explainer = shapley(Mdl,meas,'QueryPoints',data,...
    NumObservationsToSample='all',UseParallel=true,Method='conditional'); % interventional/conditional

tbl = explainer.MeanAbsoluteShapley;

figure; set(gcf,'Position', [300 400 800 250])
bar(0.03:0.03:3,tbl{:,2:6},"stacked")
legend('W1','W2','W3','W4','W5') % Mdl.ClassNames
ylabel('mean |SHAP|'); xlabel('Features (frequencies at time indexes)')

% figure; swarmchart(explainer,NumImportantPredictors=10,ColorMap='bluered')

