clear all;

% add dependencies
addpath(DEF_PATH.LIBLINEAR_PATH);
addpath(DEF_PATH.DOMAIN_TRANFORM_ECCV10);
addpath('./MMDT/');

amazon = 1; webcam = 2; dslr = 3; caltech = 4;
source_domain  = 1;
target_domain  = 3;
param = config_office(source_domain, target_domain);

% load data
[Data, Labels] = load_data_office(param.DATA_DIR, param.norm_type);

% Store results:
n = param.num_trials;
telapsed_src     = zeros(n,1);
accuracy_src     = zeros(n,1);
telapsed_tar     = zeros(n,1);
accuracy_tar     = zeros(n,1);
telapsed_asvm    = zeros(n,1);
accuracy_asvm    = zeros(n,1);
telapsed_pmt_svm = zeros(n,1);
accuracy_pmt_svm = zeros(n,1);
telapsed_src_ssvm     = zeros(n,1);
accuracy_src_ssvm     = zeros(n,1);
telapsed_tar_ssvm     = zeros(n,1);
accuracy_tar_ssvm     = zeros(n,1);

for i = 1 : n
    
    % Load data splits
    [train_ids, test_ids] = load_splits(source_domain, target_domain, param);
    data.train.source = Data{source_domain}(train_ids.source{i}, :);
    data.train.target = Data{target_domain}(train_ids.target{i}, :);
    data.test.target  = Data{target_domain}(test_ids.target{i}, :);
    
    labels.train.source = Labels{source_domain}(train_ids.source{i});
    labels.train.target = Labels{target_domain}(train_ids.target{i});
    labels.test.target  = Labels{target_domain}(test_ids.target{i});
    labels = update_labels(labels, param);
    
    if param.dim < size(data.train.source, 2)
        P = princomp([data.train.source; data.train.target; data.test.target]);
        data.train.source = data.train.source * P(:, 1:param.dim);
        data.train.target = data.train.target * P(:, 1:param.dim);
        data.test.target = data.test.target * P(:, 1:param.dim);
    end
    
    % Target domain classifier
    tstart = tic;
    model_tar = Train(labels.train, data.train, param, target_domain);
    telapsed = toc(tstart);
    
    [accuracy_tar(i), pl1, scores, confus, acc_overall] = test_svm(model_tar, labels.test.target, data.test.target);
    [pl, accuracy] = predict(labels.test.target', ...
        [sparse(data.test.target), ones(length(labels.test.target),1)], ...
        model_tar.svmmodel);
    fprintf('Target domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy_tar(i), telapsed);
    
    % Source domain classifier
    tstart = tic;
    model_src = Train(labels.train, data.train, param, source_domain);
    telapsed = toc(tstart);
    [accuracy_src(i), pl1, scores, confus, acc_overall] = test_svm(model_src, labels.test.target, data.test.target);
    [pl, accuracy] = predict(labels.test.target', ...
        [sparse(data.test.target), ones(length(labels.test.target),1)], ...
        model_src.svmmodel);
    fprintf('Source domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy_src(i), telapsed);
end
% Domain adaptation
%tstart = tic;
%[model_mmdt, W] = TrainMmdt(labels.train, data.train, param);
%telapsed = toc(tstart);

% [pl, accuracy] = predict(labels.test.target', ...
%     [sparse(data.test.target), ones(length(labels.test.target),1)], ...
%     model_mmdt.svmmodel);
%accuracy = test_svm(model_mmdt, labels.test.target, data.test.target);
%fprintf('After adaptation, accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);