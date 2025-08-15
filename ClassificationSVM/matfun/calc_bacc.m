% Simple function to calculate balanced accuracy from true and predicted
% classes
% 
% IN
%   labels: true labels for a set of examples
%   preds: predicted labels by a model for the same set of examples
% 
% OUT
%   acc: accuracy 
%   bacc: balanced accuracy

function [acc,bacc] = calc_bacc(labels,preds)

% Calculate accuracy as the percentage of correctly predicted examples
acc = sum((labels - preds) == 0)/length(labels)*100;

% Balanced accuracy is the arithmetic mean of the recall calculated
% separately for each class

classes = unique(labels);
nclass = length(classes);

for ii = 1:nclass
    iv = find(labels == classes(ii));
    tmp_labels = labels(iv);
    tmp_preds = preds(iv);
    
    tp = sum(tmp_labels == tmp_preds); % True positives
    fn = sum(tmp_labels ~= tmp_preds); % False negatives
    
    rc(ii) = tp / (tp + fn); % recall

    clear tmp_* tp fn
end

bacc = sum(rc)/length(rc);
