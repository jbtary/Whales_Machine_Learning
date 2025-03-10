% Feature scaling application to a matrix. Scaling is applied independently
% for each feature (column-wise)
% 
% IN
%   data: matrix of training examples of dimensions N training examples 
%   x M features
%   flag: different scaling available (1: between -1 and 1, 2: between 0
%   and 1, 3: using zscore)
% 
% OUT
%   data_s: scaled matrix of training examples

function data_s = feat_scal(data,flag)

switch flag
    case 1 % Scaling [-1,1]
        % From LIBSVM package docu
        data_s = (data - repmat(min(data,[],1),size(data,1),1))*...
            spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
        data_s = data_s.*2 - 1;
    case 2 % Scaling [0,1]
        % From LIBSVM package docu
        data_s = (data - repmat(min(data,[],1),size(data,1),1))*...
            spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
    case 3 % Scaling using zscore
        data_s = zscore(data,0,1);
    otherwise
        disp('Choose between 1, 2 and 3 for flag')
        data_s = [];
end

