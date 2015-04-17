function da_scene_aug()
% Feature augmentation

clear all;

% add dependencies
addpath('../liblinear-mmdt/matlab/');
addpath(genpath('./H-ASVMs/'));
addpath('./def/');

cache_dir = './cache/AUG/';
exists_or_mkdir(cache_dir);

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

data_aug = augment_data(data);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

%------------------------------------------------------------------------------
% Domain adaptation with feature augmentation
telapsed_aug = 0;
try
    load([cache_dir 'model_aug.mat']);
catch
    tstart = tic;
    model_aug = train_svm_aug(param, data_aug, labels);
    telapsed_aug = toc(tstart);
    save([cache_dir 'model_aug.mat'], 'model_aug');
end

% Test on training samples
[ovr_acc.train.source, avg_acc.train.source, ~, scores.train.source] =...
    test_svm(model_aug, labels.train.source, data_aug.train.source);

[ovr_acc.train.target, avg_acc.train.target, ~, scores.train.target] =...
    test_svm(model_aug, labels.train.target, data_aug.train.target);

% Test on testing samples
[ovr_acc.test.target, avg_acc.test.target, ~, scores.test.target] =...
    test_svm(model_aug, labels.test.target, data_aug.test.target);

writeScoreTxt(scores, param, cache_dir, model_aug, true);

% writeTransformedTARfeature(W, param, cache_dir);

fprintf(['AUG average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], ...
    avg_acc.test.target,...
    ovr_acc.test.target,...
    telapsed_aug);
end

function data_aug = augment_data(data)
% ft_src = [x, x, 0]
% ft_tar = [x, 0, x]
dim = size(data.train.source);
data_aug.train.source = [data.train.source, data.train.source, zeros(dim)];

dim = size(data.train.target);
data_aug.train.target = [data.train.target, zeros(dim), data.train.target];

dim = size(data.test.target);
data_aug.test.target = [data.test.target, zeros(dim), data.test.target];
end

function model = train_svm_aug(param, data_aug, labels)

if ~isfield(param, 'C_s') || ~isfield(param, 'C_t')
    param.C_s = 1;
    param.C_t = 1;
end

X = [data_aug.train.source; data_aug.train.target];
Y = [labels.train.source, labels.train.target];

data_svm = AugmentWithOnes(param, X);
labels_svm = Y;
param.weights = param.C_t * ones(length(Y), 1);
param.svm.C = param.C_t;

svm = train(param.weights, labels_svm', sparse(data_svm),...
    sprintf('-B %f -c %f -q', param.svm.biasMultiplier, param.svm.C)) ;

w = svm.w' ;
model.svmmodel = svm;
model.b = param.svm.biasMultiplier * w(end, :) ;
model.w = w(1:end-1, :);
model.Label = svm.Label;
end

function aug_data = AugmentWithOnes(param, data)
if param.svm.biasMultiplier == -1
    aug_data = [data, ones(size(data,1),1)];
else
    aug_data = data;
end
end