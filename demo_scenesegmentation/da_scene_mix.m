function da_scene_mix()
% Mix source and target samples

clear all;

% add dependencies
addpath('../liblinear-mmdt/matlab/');
addpath(genpath('./H-ASVMs/'));
addpath('./def/');

cache_dir = './cache/MIX/';
exists_or_mkdir(cache_dir);

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

%------------------------------------------------------------------------------
% Domain adaptation with feature augmentation
telapsed_mix = 0;
try
    load([cache_dir 'model_mix.mat']);
catch
    tstart = tic;
    model_mix = train_svm_mix(param, data, labels);
    telapsed_mix = toc(tstart);
    save([cache_dir 'model_mix.mat'], 'model_mix');
end

% Test on training samples
[ovr_acc.train.source, avg_acc.train.source, ~, scores.train.source] =...
    test_svm(model_mix, labels.train.source, data.train.source);

[ovr_acc.train.target, avg_acc.train.target, ~, scores.train.target] =...
    test_svm(model_mix, labels.train.target, data.train.target);

% Test on testing samples
[ovr_acc.test.target, avg_acc.test.target, ~, scores.test.target] =...
    test_svm(model_mix, labels.test.target, data.test.target);

writeScoreTxt(scores, param, cache_dir, model_mix);

% writeTransformedTARfeature(W, param, cache_dir);

fprintf(['AUG average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], ...
    avg_acc.test.target,...
    ovr_acc.test.target,...
    telapsed_mix);
end

function model = train_svm_mix(param, data, labels)

if ~isfield(param, 'C_s') || ~isfield(param, 'C_t')
    param.C_s = 1;
    param.C_t = 1;
end

X = [data.train.source; data.train.target];
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