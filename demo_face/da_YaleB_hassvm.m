% Test H-ASVM
clear all;

% add dependencies
addpath('./demo_face/');
addpath(genpath('./external/minConf/'));
addpath('./def/');
addpath(genpath('./H-ASVMs/'));

param = config_face_YaleB();

% Hierarchy definition, for SSVM
param.DEF_MODEL_IDS = DEF_MODEL_IDS_2L;
param.model_define = @(m, C) mt_hasvm_model_defines(m, C, 2,...
    param.DEF_MODEL_IDS, 2);

% Load data
[data, labels] = loadYaleBDataGist(param);

cache_dir = param.cache_dir;

test_domain = 1;
switch test_domain
    case 1
        data.test.target = data.test.t1;
        labels.test.target = labels.test.t1;
    case 2
        data.test.target = data.test.t2;
        labels.test.target = labels.test.t2;
end
fprintf('Testing domain = %d\n', test_domain);

target_domains = 1:2;
% For each testing domain
t = 1;
data.train.targets{t}   = data.train.t1;
mid = get_model_id(target_domains(t), target_domains, param);
data.train.targets_model_ids{t}  = mid*ones(1,size(data.train.targets{t},1));
labels.train.targets{t} = labels.train.t1;

t = 2;
data.train.targets{t}   = data.train.t2;
mid = get_model_id(target_domains(t), target_domains, param);
data.train.targets_model_ids{t}  = mid*ones(1,size(data.train.targets{t},1));
labels.train.targets{t} = labels.train.t2;


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

% H-ASVMs
telapsed = 0;
try
    load([cache_dir 'model_hasvms.mat']);
catch
    tstart = tic;
    [~, model_hasvms] = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.HASVM, model_src_ssvm);
    telapsed = toc(tstart);
    save([cache_dir 'model_hasvms.mat'], 'model_hasvms');
end

mid = get_model_id(test_domain, target_domains, param);
model_hasvm = get_hsvm_model(mid, model_hasvms);
acc = test_svm(model_hasvm, labels.test.target, data.test.target);
fprintf('H-ASSVM accuracy = %6.2f (Time = %6.2f)\n', acc, telapsed);