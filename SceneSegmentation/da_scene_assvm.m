clear all;

% add dependencies
addpath('./SceneSegmentation/');
addpath(genpath('./external/minConf/'));
addpath('./def/');
addpath(genpath('./H-ASVMs/'));

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);

% Hierarchy definition, for SSVM
param.DEF_MODEL_IDS = DEF_MODEL_IDS_2L;
param.model_define = @(m, C) mt_hasvm_model_defines(m, C, 2, param.DEF_MODEL_IDS);

% Load data
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

if param.dim < size(data.train.source, 2)
    P = princomp([data.train.source; data.train.target; data.test.target]);
    data.train.source = data.train.source * P(:, 1:param.dim);
    data.train.target = data.train.target * P(:, 1:param.dim);
    data.test.target = data.test.target * P(:, 1:param.dim);
end

% Source domain classifier
tstart = tic;
model_src_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
telapsed = toc(tstart);
accuracy = test_svm(model_src_ssvm, labels.test.target, data.test.target, param);
fprintf('Source domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% Target domain classifier
tstart = tic;
model_tar_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR_SSVM);
telapsed = toc(tstart);
accuracy = test_svm(model_tar_ssvm, labels.test.target, data.test.target, param);
fprintf('Target domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% ASSVM
tstart = tic;
model_assvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.ASSVM, model_src_ssvm);
telapsed = toc(tstart);
accuracy = test_svm(model_assvm, labels.test.target, data.test.target, param);
fprintf('After adaptation, accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);