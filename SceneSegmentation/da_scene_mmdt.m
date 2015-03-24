clear all;

% add dependencies
addpath('../liblinear-mmdt/matlab/');
addpath('./external/DomainTransformsECCV10/');
addpath('./SceneSegmentation/');
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
[avg_acc.train.target, ~, scores.train.target, ~, ovr_acc.train.target] =...
    test_svm(model_tar, labels.train.target, data.train.target);
% Test on testing samples
[avg_acc.test.target, ~, scores.test.target, ~, ovr_acc.test.target] =...
    test_svm(model_tar, labels.test.target, data.test.target);
save([cache_dir 'scores_tar.mat'], 'scores');
% writeScoreTxt(scores, param, cache_dir);

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

[acc_src, ~, ~, ~, acc_overall_src] = test_svm(model_src, labels.test.target, data.test.target);

fprintf(['SRC average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_src, acc_overall_src, telapsed_src);

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

% Test on training samples
[avg_acc.train.source, ~, scores.train.source, ~, ovr_acc.train.source] =...
    test_svm(model_mmdt, labels.train.source, data.train.source);
[avg_acc.train.target, ~, scores.train.target, ~, ovr_acc.train.target] =...
    test_svm(model_mmdt, labels.train.target, data.train.target);
% Test on testing samples
[avg_acc.test.target, ~, scores.test.target, conf, ovr_acc.test.target] =...
    test_svm(model_mmdt, labels.test.target, data.test.target);
save([cache_dir 'scores_mmdt.mat'], 'scores');
% writeScoreTxt(scores, param, cache_dir);
writeTransformedSRCfeature(W, param, cache_dir);

fprintf(['MMDT average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], ...
    avg_acc.test.target,...
    ovr_acc.test.target,...
    telapsed_mmdt);