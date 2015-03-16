clear all;

% add dependencies
addpath('./SceneSegmentation/');
addpath('./def/');
addpath(genpath('./external/'));

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);

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

% SRC (SVM)
tstart = tic;
model_src = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC);
telapsed = toc(tstart);
accuracy = test_svm(model_src, labels.test.target, data.test.target, param);
fprintf('   SRC: Accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% TAR (SVM)
tstart = tic;
model_tar = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR);
telapsed = toc(tstart);
accuracy = test_svm(model_tar, labels.test.target, data.test.target, param);
fprintf('   TAR: Accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% A-SVM
tstart = tic;
model_asvm = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.ASVM, model_src);
telapsed = toc(tstart);
accuracy = test_svm(model_asvm, labels.test.target, data.test.target, param);
fprintf('   ASVM: Accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% PMT-SVM
tstart = tic;
model_pmt_svm = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.PMT_SVM, model_src);
telapsed = toc(tstart);
accuracy = test_svm(model_pmt_svm, labels.test.target, data.test.target, param);
fprintf('   PMT-SVM: Accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);