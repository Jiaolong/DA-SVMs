% Author jhoffman@eecs.berkeley.edu (Judy Hoffman)
% Implementation code for the paper:
% "Efficient Learning of Domain-invariant Image Representations"
% J. Hoffman, E. Rodner, J. Donahue, K. Saenko, T. Darrell
% International Conference on Learning Representations (ICLR), 2013.
function [model, A] = TrainMmdt(labels, data, param)
% Usage:
%   [model, A] = TrainMmdt(labels, data, param)
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
if ~isfield(param, 'gamma')
    param.gamma = 10^(-4);
end

dA = size(data.source,2);
dB = size(data.target,2);
param.A = eye(dB+1, dA+1);

if ~isfield(param, 'train_classes')
    param.train_classes = sort(unique(labels.source));
end
% Iterate between learning SVMs and transforms
for iter = 1:param.mmdt_iter
    [model, data, param] = TrainMmdtOneIter(labels, data, param, iter);
end
model.w = model.w * param.A';
A = param.A;
end

function [model, data, param] = TrainMmdtOneIter(labels, data, param, iter)

data.transformed_target = AugmentWithOnes(data.target)*param.A;
data_svm = [AugmentWithOnes(data.source); data.transformed_target];
labels_svm = [labels.source, labels.target];

weights_s = param.C_s * ones(length(labels.source), 1);
weights_t = param.C_t * ones(length(labels.target), 1);
param.weights = [weights_s; weights_t];
if iter == 1 && isfield(param, 'source_svm')
    model = param.source_svm;
else
    model = train(param.weights, labels_svm', sparse(data_svm), '-c 1 -q');
end

tstart = tic;
L = learnAsymmTransformWithSVM(model.w(param.train_classes,:), ...
    param.train_classes, AugmentWithOnes(data.target), labels.target, param);
param.A = L';
param.telapsed(iter) = toc(tstart);
end

function aug_data = AugmentWithOnes(data)
aug_data = [data, ones(size(data,1),1)];
end