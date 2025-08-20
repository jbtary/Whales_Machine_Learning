%% Calculate the evaluation metrics for model using predictions and labels
% Takes 20-30 min to calculate
clear

Fs = 100; % Sampling frequency (Hz)
delay = 0.1*Fs; % Time separation to define "discontinuous" parts in samples
minwin = 3*Fs; % Minimum duration of windows in samples
prop = 0.25; % Proportion of detection windows to call a detection

thr = [0.1:0.1:0.8 0.85:0.01:1]; % Threshold on detection function
p_pic = [0.1:0.1:0.8 0.85:0.01:1]; % Threshold on P picking function

% All threshold combinations: first column is threshold detection, 2nd
% column is threshold P pick
[n,m] = ndgrid(thr',p_pic');
combi = [m(:),n(:)]; clear n m

% P pred, Det pred, Det true for all data examples of the eval dataset
examples = dir('EvalMetrics/*.dat');

% Elements for confusion matrix, 3 metrics: TP, FP, FN, all examples, all
% threshold combinations
conf = zeros(length(combi),length(examples),3); 

for ii = 1:length(examples)
    % Event characteristic functions: P pred, Det pred, Det true
    % ev = load(['EvalMetrics/Ev' num2str(ii) '.dat']); % <- not used, for testing
    ev = load(['EvalMetrics/' examples(ii).name]);
    %figure; plot(1:length(ev),ev(:,1),1:length(ev),ev(:,2),1:length(ev),ev(:,3)); legend('P pred','Det pred','Det true')
    
    % Determine if it is a 0 (no whale call) or a 1 (whale call present)
    if sum(ev(:,3)) > 0
        wc(ii,1) = 1;
    else
        wc(ii,1) = 0;
    end
    
    for cc = 1:length(combi)
        % Determine if this is a TP, FP, or a FN
        % 1st step: determine if there is a detection
        % Definition of a detection: Det pred > threshold1 for a minimum amount
        % of continuous time samples + P pred > threshold2
        idx = find(ev(:,2) > combi(cc,1)); % Find all samples > threshold1
        if ~isempty(idx)
            der = idx(1:end-1) - idx(2:end); % Kind of a derivative (to find peaks corresponding to large time differences)

            idx2 = find(abs(der) > round(delay)); % Consider continuous gaps up to delay
            if isempty(idx2) % In case there is only one window
                win = [idx(1) idx(end)];
            else
                for jj = 1:length(idx2) % Definition of the trigger and detrigger times
                    if jj == 1
                        win(jj,:) = [idx(1) idx(idx2(jj))];
                    else
                        win(jj,:) = [idx(idx2(jj-1)+1) idx(idx2(jj))];
                    end
                end
                % Windows of trigger and detrigger times
                win(jj+1,:) = [idx(idx2(jj)+1) idx(end)];
            end

            % Clean windows that are too short
            win(win(:,2) - win(:,1) < minwin,:) = [];

            % Detection or not?
            if ~isempty(win)
                if size(win,1) > 1
                    % Select the window with the highest Det pred and the
                    % longest
                    for jj = 1:size(win,1)
                        maxwin(jj,1) = max(ev(win(jj,1):win(jj,2),2));
                    end

                    % Check for which one(s) is the max
                    idx3 = find(maxwin == max(maxwin));

                    if length(idx3) > 1 % If more than one same max, check length of windows
                        winlen = win(:,2) - win(:,1);
                        % It would be very unlucky to have the same max and
                        % same length (did not happen)
                        idx3 = find(maxwin == max(maxwin) & winlen == max(winlen));
                    end

                    win = win(idx3,:); % Select the window
                    % Window selection could be changed following 2 options:
                    % 1-select the window with max P pick value
                    % 2-do not select window and check them all wrt true
                    % detection function
                end

                det = 0;
                begt = win(1,1) - (1*Fs); endt = win(1,2);
                if begt <= 0; begt = 1; end
                if endt > length(ev); endt = length(ev); end
                if max(ev(begt:endt,1)) >= combi(cc,2) % Check P detection threshold
                    det = 1;
                end
            else
                det = 0; % No window = no detection
            end
        else
            det = 0; % Nothing above threshold = no detection
        end

        % 2nd step: check if the predicted detection window (if any) is at 
        % the same spot as the true detection position
        if wc(ii,1) == 1 && det == 1
            fake_pred = zeros(length(ev),1);
            fake_pred(win(1,1):win(1,2)) = 1;

            overlap = fake_pred == 1 & ev(:,3) == 1;
            % Say that if the overlap is less than prop(%) then it's not ok
            if sum(overlap) < prop*sum(ev(:,3))
                det = 0;
            end
        end

        % Update the evaluation metrics
        if wc(ii,1) == 1 && det == 1 % TP
            conf(cc,ii,1) = 1;
        elseif wc(ii,1) == 1 && det == 0 % FN
            conf(cc,ii,3) = 1;
        elseif wc(ii,1) == 0 && det == 1 % FP
            conf(cc,ii,2) = 1;
        end

        clear idx* det der win begt endt maxwin winlen overlap fake_pred
    end

    clear ev
end

% save('eval_metrics_15112024_2',...
%     'combi','conf','delay','examples','Fs','minwin','p_pic','thr','wc','prop')

%% Calculate precision, recall, f1-score, auc from evaluation metrics
clear
load('eval_metrics_15112024_2')

for ii = 1:length(combi)
    tmp_conf = squeeze(conf(ii,:,:)); % TP, FP, FN
    TP = sum(tmp_conf(:,1)); % Total number of TP in eval dataset
    FP = sum(tmp_conf(:,2)); % Total number of FP in eval dataset
    FN = sum(tmp_conf(:,3)); % Total number of FN in eval dataset
    TN = sum(sum(tmp_conf,2) == 0); % Total number of TN in eval dataset
    
    if TP+FP+FN+TN ~= length(examples)
        disp('Problem, condition cases don t sum up to number of examples')
    end
    
    precision(ii,1) = TP / (TP + FP);
    recall(ii,1) = TP / (TP + FN);
    f1score(ii,1) = (2*precision(ii,1)*recall(ii,1))/(precision(ii,1)+recall(ii,1));
    clear TP FP FN TN tmp_conf
end

% Plot the recall vs precision
figure; set(gcf,'Position', [400 200 400 300])
scatter(recall*100,precision*100,exp(5*combi(:,1)),combi(:,2))
xlabel('Recall (%)'); ylabel('Precision (%)'); colormap('jet'); colorbar; box on
%prettyfier(14,'SansSerif',1,0)
set(gca,'FontSize',14)
set(gca,'XTick',0:20:100); set(gca,'YTick',0:20:100);
axis([0 100 0 100])

% Find the best combination of threshold parameters
[maxf1,iloc] = max(f1score);
maxprecision = precision(iloc);
maxrecall = recall(iloc);
max_thr = combi(iloc,1);
max_p_pic = combi(iloc,2);

% Calculate the AUC for the (best) threshold
idxnan = ~isnan(precision); % Some nan in precision but not in recall
tmp_recall = recall(idxnan);
tmp_precision = precision(idxnan);
tmp_combi = combi(idxnan,1);
[sort_recall,sort_idx] = sort(tmp_recall);
sort_precision = tmp_precision(sort_idx);
sort_combi = tmp_combi(sort_idx);
auc = trapz(sort_recall,sort_precision);
% iv = find(sort_combi == max_thr);
% auc = trapz(sort_recall(iv),sort_precision(iv)); % Using only best threshold

% tmp_conf = squeeze(conf(iloc,:,:)); % TP, FP, FN
% TP = sum(tmp_conf(:,1)); % Total number of TP in eval dataset
% FP = sum(tmp_conf(:,2)); % Total number of FP in eval dataset
% FN = sum(tmp_conf(:,3)); % Total number of FN in eval dataset
% TN = sum(sum(tmp_conf,2) == 0); % Total number of TN in eval dataset
% FPR = FP / (FP + TN);

%% 2D plots of precision, recall and f1-score depending on the two thresholds

prec_nan = precision; prec_nan(isnan(precision)) = 0;
f1_nan = f1score; f1_nan(isnan(f1_nan)) = 0;

figure; subplot(131)
scatter(combi(:,1),combi(:,2),25,prec_nan,'filled')
axis square; xlabel('Detection threshold'); ylabel('P picking threshold')
clim([0.7 1]); title('Precision')

subplot(132)
scatter(combi(:,1),combi(:,2),25,recall,'filled')
axis square; xlabel('Detection threshold'); ylabel('P picking threshold')
clim([0.7 1]); title('Recall')

subplot(133)
scatter(combi(:,1),combi(:,2),25,f1_nan,'filled')
axis square; xlabel('Detection threshold'); ylabel('P picking threshold')
clim([0.7 1]); title('f1-score'); colorbar

clear *_nan
