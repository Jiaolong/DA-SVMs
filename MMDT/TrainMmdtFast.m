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

[model, A] = train_linear_mmdt_fast(weights_t,...
    labels.target', sparse(data.target'), weights_s, labels.source', sparse(data.source'));
end