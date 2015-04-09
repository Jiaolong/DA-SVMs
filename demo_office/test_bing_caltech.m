% Demo experiment: domain adaptation from Bing to Caltech

clear all;
addpath util;

param = config_bing_experiment();
addpath(param.ECCV2010_releasePath);
addpath(param.LIBSVM_path);

%--------------------------------------------------------------------------
%{{{ load data
dataset_dir = param.DATASET_path;
if isempty(dataset_dir)
    fprintf('Please set the directory path for the bing-caltech dataset\n');
    return;
end

categories = param.categories;

%{{ Load Source Domain: caltech256
disp('Loading CalTech256 Data');
load(fullfile(dataset_dir, sprintf(param.fileName_c256, param.nttuse_c256)));
fstar = double(real(fstar > 0));

ind_tr = tr_label <= numel(param.categories);
data.train.source = NormalizeData(fstar(:,ind_tr)', param.norm); 
labels.train.source = tr_label(ind_tr)';

clear fstar; clear tr_label;
clear fstar_test; clear te_label;
%}}

%{{ Load Target Domain: Bing
disp('Loading Bing Data');
load(fullfile(dataset_dir, sprintf(param.fileName_bing, param.nttuse_bing)));
fstar = double(real(fstar>0));

ind_tr = tr_label <= numel(param.categories);
data.train.target = NormalizeData(fstar(:,ind_tr)', param.norm);
labels.train.target = tr_label(ind_tr)';

ind_te = te_label <= numel(param.categories);
fstar_test = double(real(fstar_test > 0));
data.test.target = NormalizeData(fstar_test(:,ind_te)', param.norm);
labels.test.target = te_label(ind_te)';

clear fstar; clear tr_label;
clear fstar_test; clear te_label;
%}}
%}}}

%--------------------------------------------------------------------------
% SRC Knn - No adaptation no multi-domain baseline
tstart = tic;
[a, p] = KernelKnnClassify(labels.train.source, data.train.source, ...
    labels.test.target, data.test.target);
accuracy_knn = a;
telapsed_knn = toc(tstart);
fprintf('SRC-KNN: Accuracy = %6.2f (Time = %6.2f)\n', accuracy_knn, telapsed_knn);

% SRC SSVM - No adaptation no multi-domain baseline
tstart = tic;
model_src_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
telapsed_ssvm = toc(tstart);
acc_ssvm = test_svm(model_src_ssvm, labels.test.target, data.test.target);
accuracy_ssvm = acc_ssvm;
fprintf('SRC-SSVM: Accuracy = %6.2f (Time = %6.2f)\n', accuracy_ssvm, telapsed_ssvm);

% ASSVM
% tstart = tic;
% model_assvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.ASSVM, model_src_ssvm);
% telapsed_assvm = toc(tstart);
% acc_assvm = test_svm(model_assvm, labels.test.target, data.test.target);
% accuracy_assvm = acc_assvm;
% fprintf('A-SSVM: Accuracy = %6.2f (Time = %6.2f)\n', accuracy_assvm, telapsed_assvm);

%--------------------------------------------------------------------------
% Cluster the target data into sub-domains
tstart = tic;
fprintf('Domain Clustering into %d Domains . . .\n', param.num_clusters);
target_subdomains = DomainDiscovery ([data.train.target; data.test.target], ...
    [labels.train.target labels.test.target], param.num_clusters, [], param);
telapsed_cluster = toc(tstart);
fprintf('    Finished learning clusters in %6.2f s\n',telapsed_cluster);

for i=1:size(data.train.target,1)
    m_id =  DEF_MODEL_IDS.M_T1 + target_subdomains(i) - 1;
    assert(m_id <= DEF_MODEL_IDS.M_T3);
    data.train.target_model_ids(i) = m_id;
end
test_target_subdomains = DEF_MODEL_IDS.M_T1 + ...
    target_subdomains(size(data.train.target,1)+1:end)-1;
data.train.targets = data.train.target;
labels.train.targets = labels.train.target;

%--------------------------------------------------------------------------
% H-ASVMs
tstart = tic;
[~, models_hasvm] = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.HASVM, model_src_ssvm, []);
telapsed_hasvm = toc(tstart);

model_test_ids = [DEF_MODEL_IDS.M_T1, DEF_MODEL_IDS.M_T2];
%model_test_ids = [DEF_MODEL_IDS.M_T1, DEF_MODEL_IDS.M_T2, DEF_MODEL_IDS.M_T3];
for model_test_id = model_test_ids
    model = models_hasvm{model_test_id - DEF_MODEL_IDS.M_T1 + 3};
    test_ids = 1:size(data.test.target, 1);
    test_ids = test_ids(test_target_subdomains == model_test_id);
    acc_hasvm = test_svm(model, labels.test.target(test_ids), data.test.target(test_ids,:));
    accuracy_hasvm = acc_hasvm;
    fprintf('H-ASSVM: Sub-domain(%d), Accuracy = %6.2f (Time = %6.2f)\n', model_test_id, accuracy_hasvm, telapsed_hasvm);
end