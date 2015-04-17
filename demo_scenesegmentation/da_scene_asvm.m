clear all;

% add dependencies
addpath('./def/');
addpath(genpath('./H-ASVMs/'));
addpath('../liblinear-mmdt/matlab/');
addpath(genpath('./external/'));

cache_dir = './cache/ASVM/';
exists_or_mkdir(cache_dir);

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

%------------------------------------------------------------------------------
% TAR (SVM)
telapsed_tar = 0;
try
    load([cache_dir 'model_asvmlinear_tar.mat']);
catch
    tstart = tic;
    model_tar = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR);
    telapsed_tar = toc(tstart);
    save([cache_dir 'model_asvmlinear_tar.mat'], 'model_tar');
end

% Test on training samples
[ovr_acc.train.target, avg_acc.train.target, ~, scores.train.target] =...
    test_svm(model_tar, labels.train.target, data.train.target);
% Test on testing samples
[ovr_acc.test.target, avg_acc.test.target, ~, scores.test.target] =...
    test_svm(model_tar, labels.test.target, data.test.target);

fprintf(['TAR average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], ...
    avg_acc.test.target,...
    ovr_acc.test.target,...
    telapsed_tar);

%------------------------------------------------------------------------------
% SRC (SVM)
telapsed_src = 0;
try
    load([cache_dir 'model_asvmlinear_src.mat']);
catch
    tstart = tic;
    model_src = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC);
    telapsed_src = toc(tstart);
    save([cache_dir 'model_asvmlinear_src.mat'], 'model_src');
end

% Test on training samples
[ovr_acc.train.source, avg_acc.train.source, ~, scores.train.source] =...
    test_svm(model_src, labels.train.source, data.train.source);
% Test on testing samples
[ovr_acc.test.target, avg_acc.test.target, ~, scores.test.target] =...
    test_svm(model_src, labels.test.target, data.test.target);

fprintf(['SRC average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], ...
    avg_acc.test.target,...
    ovr_acc.test.target,...
    telapsed_src);

%writeScoreTxt(scores, param, cache_dir, model_src);

%------------------------------------------------------------------------------
% A-SVM
telapsed_asvm = 0;
try
    load([cache_dir 'model_asvmlinear_asvm.mat']);
catch
    load([cache_dir 'model_asvmlinear_src.mat']);
    tstart = tic;
    model_asvm = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.ASVM, model_src);
    telapsed_asvm = toc(tstart);
    save([cache_dir 'model_asvmlinear_asvm.mat'], 'model_asvm');
end

% Test on training samples
[ovr_acc.train.source, avg_acc.train.source, ~, scores.train.source] =...
    test_svm(model_asvm, labels.train.source, data.train.source);
[ovr_acc.train.target, avg_acc.train.target, ~, scores.train.target] =...
    test_svm(model_asvm, labels.train.target, data.train.target);
% Test on testing samples
[ovr_acc.test.target, avg_acc.test.target, ~, scores.test.target, conf] =...
    test_svm(model_asvm, labels.test.target, data.test.target);

writeScoreTxt(scores, param, cache_dir, model_asvm);

fprintf(['ASVM average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], ...
    avg_acc.test.target,...
    ovr_acc.test.target,...
    telapsed_asvm);
