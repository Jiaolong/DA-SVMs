clear all;

% add dependencies
addpath('./SceneSegmentation/');
addpath('./def/');
addpath(genpath('./H-ASVMs/'));
addpath('../liblinear-mmdt/matlab/');
addpath(genpath('./external/'));

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

% TAR (SVM)
tstart = tic;
model_tar = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR);
telapsed_tar = toc(tstart);
save('cache/model_asvmlinear_tar.mat', 'model_tar');

[acc_tar, ~, ~, ~, acc_overall_tar] = test_svm(model_tar, labels.test.target, data.test.target);
fprintf(['TAR average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_tar, acc_overall_tar, telapsed_tar);

% SRC (SVM)
tstart = tic;
model_src = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC);
telapsed_src = toc(tstart);
save('cache/model_asvmlinear_src.mat', 'model_src');

[acc_src, ~, ~, ~, acc_overall_src] = test_svm(model_src, labels.test.target, data.test.target);
fprintf(['SRC average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_src, acc_overall_src, telapsed_src);
    
% A-SVM
tstart = tic;
model_asvm = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.ASVM, model_src);
telapsed_asvm = toc(tstart);
save('cache/model_asvmlinear_asvm.mat', 'model_mmdt');
[acc_asvm, ~, ~, ~, acc_overall_asvm] = test_svm(model_asvm, labels.test.target, data.test.target);

fprintf(['ASVM average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_asvm, acc_overall_asvm, telapsed_asvm);
