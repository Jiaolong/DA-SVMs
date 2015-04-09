clear all;

% add dependencies
addpath('./demo_face/');
addpath(genpath('./external/minConf/'));
addpath('./def/');
addpath(genpath('./H-ASVMs/'));

param = config_face_icpr04();

% Hierarchy definition, for SSVM
param.DEF_MODEL_IDS = DEF_MODEL_IDS_2L;
param.model_define = @(m, C) mt_hasvm_model_defines(m, C, 2, param.DEF_MODEL_IDS);

% Load data
[data, labels] = loadICPR04DataGist(param);

source_domain = param.source;
target_domain = param.target;
cache_dir = param.cache_dir;


test_domain = 4;
TRAIN_TAR_ALL = false;
fprintf('ASSVM: Test domain = %d\n', test_domain);
switch test_domain
    case 1
        labels.test.target = labels.test.t1;
        data.test.target = data.test.t1;
        data.train.target = data.train.t1;
        labels.train.target = labels.train.t1;
    case 2
        labels.test.target = labels.test.t2;
        data.test.target = data.test.t2;
        data.train.target = data.train.t2;
        labels.train.target = labels.train.t2;
    case 3
        labels.test.target = labels.test.t3;
        data.test.target = data.test.t3;
        data.train.target = data.train.t3;
        labels.train.target = labels.train.t3;
    case 4
        labels.test.target = labels.test.t4;
        data.test.target = data.test.t4;
        data.train.target = data.train.t4;
        labels.train.target = labels.train.t4;
end

if TRAIN_TAR_ALL % train with all target domains
    labels.train.target = [labels.train.t1; labels.train.t2;...
        labels.train.t3; labels.train.t4];
    data.train.target = [data.train.t1; data.train.t2;...
        data.train.t3; data.train.t4];
end

% Source domain classifier
telapsed = 0;
try
    load([cache_dir 'model_src_ssvm.mat']);
catch
    tstart = tic;
    model_src_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
    telapsed = toc(tstart);
    save([cache_dir 'model_src_ssvm.mat'], 'model_src_ssvm');
end
accuracy = test_svm(model_src_ssvm, labels.test.target, data.test.target);
fprintf('Source domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% Target domain classifier
tstart = tic;
model_tar_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR_SSVM);
telapsed = toc(tstart);
accuracy = test_svm(model_tar_ssvm, labels.test.target, data.test.target);
fprintf('Target domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% ASSVM
tstart = tic;
model_assvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.ASSVM, model_src_ssvm);
telapsed = toc(tstart);
accuracy = test_svm(model_assvm, labels.test.target, data.test.target);
fprintf('After adaptation, accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);