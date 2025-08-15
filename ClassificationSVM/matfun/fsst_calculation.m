% Example of 4th order FSST calculation and ridge extraction using the
% FSSTn toolbox
clear
%addpath(genpath('FSSTn-master/'))

% Example signal
load('../matfiles/2015_01_29_15_28_50.mat')
fmin = 10; fmax = 45; fs = 100;
data_raw = dath{21,1};
fs_ori = round(1/hdr.st(1));
data_dec = decimate(data_raw,fs_ori/fs);
[b,a]=butter(4,[fmin fmax]/(fs/2),'bandpass');
data_filt = filtfilt(b,a,data_dec);
data_fin = data_filt(901:1400);
data_fin = data_fin - mean(data_fin);
data_fin = data_fin/max(abs(data_fin));

% Transform parameters
tf.gamma = 0.001;
tf.sigma = 0.11; % value calculated from the renyi entropy

% Ridge extraction parameters
rg.clwin = 10;
rg.lambda = 0;
rg_stft.clwin = 10;
rg_stft.lambda = 0;

if mod(log2(length(data_fin)),1)~=0
    data_tmp = [data_fin zeros(1,2^(floor(log2(length(data_fin)))+1)-length(data_fin))];
end

N = length(data_tmp); % Signal length in samples
tf.ft = 1:N/2; tf.bt = 1:length(data_fin);
tf.time = (0:length(data_fin)-1)*(1/fs);
tf.freq = (0:(N-1)/2)*(fs/2)/(N/2);

% STFT,SST1,SST4
[tf.STFT,tf.SST1,tf.SST4] = sst4(data_tmp,tf.gamma,tf.sigma,tf.ft,tf.bt);
% Extract ridges for SST4
[rg.Cs1,~] = exridge_mult_Noise(tf.SST4,1,rg.lambda,rg.clwin);
[rg.Cs2,~] = exridge_mult_Noise(tf.SST4,2,rg.lambda,rg.clwin);
% Extract ridges for STFT
[rg_stft.Cs1,~] = exridge_mult_Noise(tf.STFT,1,rg_stft.lambda,rg_stft.clwin);
[rg_stft.Cs2,~] = exridge_mult_Noise(tf.STFT,2,rg_stft.lambda,rg_stft.clwin);

% Extract ridge of max amplitude for the case with 2 modes
for nn = 1:size(rg.Cs2,1)
    for jj = 1:length(tf.time)
        rg_ampl(nn,jj) = abs(tf.SST4(rg.Cs2(nn,jj),jj));
        rg_ampl2(nn,jj) = abs(tf.STFT(rg_stft.Cs2(nn,jj),jj));
    end
end

idx_sv = 1;
for jj = 1:length(tf.time)
    idx = find(rg_ampl(:,jj) == max(rg_ampl(:,jj)));
    if length(idx) > 1 % Unpretty way to handle when amplitudes are the same (use idx closest to previous one)
        [~,id] = min(abs(idx - idx_sv));
        idx = idx(id); clear id
    end
    rg.fr_fin(1,jj) = tf.freq(rg.Cs2(idx,jj));
    idx_sv = idx;
    clear idx
end

idx_sv = 1;
for jj = 1:length(tf.time)
    idx = find(rg_ampl2(:,jj) == max(rg_ampl2(:,jj)));
    if length(idx) > 1 % Unpretty way to handle when amplitudes are the same (use idx closest to previous one)
        [~,id] = min(abs(idx - idx_sv));
        idx = idx(id); clear id
    end
    rg_stft.fr_fin(1,jj) = tf.freq(rg_stft.Cs2(idx,jj));
    idx_sv = idx;
    clear idx
end
