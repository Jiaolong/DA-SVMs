function da_scene_mmdt()
clear all;

% add dependencies
addpath('../liblinear-mmdt/matlab/');
addpath('./external/DomainTransformsECCV10/');
addpath('./H-ASVMs/');
addpath('./def/');
addpath('./MMDT/');

cache_dir = './cache/MMDT/';
exists_or_mkdir(cache_dir);

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

%------------------------------------------------------------------------------
% Domain adaptation with MMDT
telapsed_mmdt = 0;
try
    load([cache_dir 'model_mmdt.mat']);
catch
    tstart = tic;
    [model_mmdt, W] = TrainMmdtFast(labels.train, data.train, param);
    telapsed_mmdt = toc(tstart);
    save([cache_dir 'model_mmdt.mat'], 'model_mmdt', 'W');
end

svm = model_mmdt.svmmodel;
model_mmdt_src.svmmodel.w = svm.w/(W');
w = model_mmdt_src.svmmodel.w';
model_mmdt_src.b = param.svm.biasMultiplier * w(end, :);
model_mmdt_src.w = w(1:end-1, :);
model_mmdt_src.Label = svm.Label;

% Test on training samples
[ovr_acc.train.source, avg_acc.train.source, ~, scores.train.source] =...
    test_svm(model_mmdt_src, labels.train.source, data.train.source);

[ovr_acc.train.target, avg_acc.train.target, ~, scores.train.target] =...
    test_svm(model_mmdt, labels.train.target, data.train.target);

% Test on testing samples
[ovr_acc.test.target, avg_acc.test.target, ~, scores.test.target] =...
    test_svm(model_mmdt, labels.test.target, data.test.target);

writeScoreTxt(scores, param, cache_dir, model_mmdt);

% writeTransformedTARfeature(W, param, cache_dir);

fprintf(['MMDT average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], ...
    avg_acc.test.target,...
    ovr_acc.test.target,...
    telapsed_mmdt);

%------------------------------------------------------------------------------
% Target domain classifier
telapsed_tar = 0;
try
    load([cache_dir 'model_tar.mat']);
catch
    tstart = tic;
    model_tar = Train(labels.train, data.train, param, target_domain);
    telapsed_tar = toc(tstart);
    save([cache_dir 'model_tar.mat'], 'model_tar');
end

% Test on training samples
[ovr_acc.train.target, avg_acc.train.target, ~, scores.train.target] =...
    test_svm(model_tar, labels.train.target, data.train.target);
% Test on testing samples
[ovr_acc.test.target, avg_acc.test.target, ~, scores.test.target] =...
    test_svm(model_tar, labels.test.target, data.test.target);
%save([cache_dir 'scores_tar30.mat'], 'scores');
%writeScoreTxt(scores, param, cache_dir, model_tar);
fprintf(['TAR average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], avg_acc.test.target,...
    ovr_acc.test.target, telapsed_tar);

%------------------------------------------------------------------------------
% Source domain classifier
telapsed_src = 0;
try
    load([cache_dir 'model_src.mat']);
catch
    tstart = tic;
    model_src = Train(labels.train, data.train, param, source_domain);
    telapsed_src = toc(tstart);
    save([cache_dir 'model_src.mat'], 'model_src');
end

[acc_overall_src, acc_src] = test_svm(model_src, labels.test.target, data.test.target);

fprintf(['SRC average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_src, acc_overall_src, telapsed_src);
end