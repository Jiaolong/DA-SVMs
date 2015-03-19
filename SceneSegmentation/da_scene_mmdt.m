clear all;

% add dependencies
addpath('../liblinear-mmdt/matlab/');
addpath('./external/DomainTransformsECCV10/');
addpath('./SceneSegmentation/');
addpath('./def/');
addpath('./MMDT/');

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

% Target domain classifier
tstart = tic;
model_tar = Train(labels.train, data.train, param, target_domain);
telapsed_tar = toc(tstart);
save('cache/model_tar.mat', 'model_tar');

[acc_tar, ~, ~, ~, acc_overall_tar] = test_svm(model_tar, labels.test.target, data.test.target);

fprintf(['TAR average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_tar, acc_overall_tar, telapsed_tar);

% Source domain classifier
tstart = tic;
model_src = Train(labels.train, data.train, param, source_domain);
telapsed_src = toc(tstart);
save('cache/model_src.mat', 'model_src');

[acc_src, ~, ~, ~, acc_overall_src] = test_svm(model_src, labels.test.target, data.test.target);

fprintf(['SRC average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_src, acc_overall_src, telapsed_src);

% Domain adaptation
tstart = tic;
[model_mmdt, W] = TrainMmdtFast(labels.train, data.train, param);
telapsed_mmdt = toc(tstart);
save('cache/model_mmdt.mat', 'model_mmdt');
[acc_mmdt, ~, ~, ~, acc_overall_mmdt] = test_svm(model_mmdt, labels.test.target, data.test.target);

fprintf(['MMDT average accuracy = %6.2f,'...
    'overall accuracy = %6.2f, (Time = %6.2f)\n'], acc_mmdt, acc_overall_mmdt, telapsed_mmdt);