function [model, A] = TrainMmdtFast(labels, data, param)
% Usage:
%   [model, A] = TrainMmdtFast(labels, data, param)
% Input:
%   labels.source, labels.target: label vectors for training data points
%   data.source, data.target: training data of the form - num_pts x num_features
%   param.C_s, param.C_t, param.mmdt_iter, param.train_classes (for use in
%   new category experiment setting)
% Output:
%   model - struct of the form in liblinear
%   A - transformation matrix learned
if ~isfield(param, 'C_s') || ~isfield(param, 'C_t')
    param.C_s = 1;
    param.C_t = 1;
end

weights_s = param.C_s * ones(length(labels.source),1);
weights_t = param.C_t * ones(length(labels.target),1);

[svm, A] = train_linear_mmdt_fast(weights_t,...
    labels.target', sparse(AugmentWithOnes(data.target)'),...
    weights_s, labels.source',...
    sparse(AugmentWithOnes(data.source)'));

w = svm.w';
model.svmmodel = svm;
model.b = param.svm.biasMultiplier * w(end, :);
model.w = w(1:end-1, :);
model.Label = svm.Label;
end

function aug_data = AugmentWithOnes(data)
aug_data = [data, ones(size(data,1),1)];
end
