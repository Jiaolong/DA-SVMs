% configuration
amazon = 1; webcam = 2; dslr = 3; caltech = 4;
source_domain  = 1; 
target_domain  = 2;
param = config(source_domain, target_domain);

% load data
[Data, Labels] = load_data(param.DATA_DIR, param.norm_type);

% Hierarchy definition, for SSVM
param.DEF_MODEL_IDS = DEF_MODEL_IDS_2L;
param.model_define = @(m, C) mt_hasvm_model_defines(m, C, 2, param.DEF_MODEL_IDS);

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

fprintf('       Iteration: %d', n);
for i = 1:n
    fprintf('...%d', n-i);
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
    
    % SRC (SVM)
    tstart = tic;
    model_src = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC);
    telapsed_src(i) = toc(tstart);
    acc_src = test_svm(model_src, labels.test.target, data.test.target);
    accuracy_src(i) = acc_src(1);
    % fprintf('   SRC: Accuracy = %6.2f (Time = %6.2f)\n', accuracy_src(i), telapsed_src(i));
    
    % TAR (SVM)
    tstart = tic;
    model_tar = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR);
    telapsed_tar(i) = toc(tstart);
    acc_tar = test_svm(model_tar, labels.test.target, data.test.target);
    accuracy_tar(i) = acc_tar(1);
    % fprintf('   TAR: Accuracy = %6.2f (Time = %6.2f)\n', accuracy_tar(i), telapsed_tar(i));
    
    % SRC (SSVM)
    tstart = tic;
    model_src_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
    telapsed_src_ssvm(i) = toc(tstart);
    acc_src_ssvm = test_svm(model_src_ssvm, labels.test.target, data.test.target);
    accuracy_src_ssvm(i) = acc_src_ssvm(1);
    
    % TAR (SSVM)
    tstart = tic;
    model_tar_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR_SSVM);
    telapsed_tar_ssvm(i) = toc(tstart);
    acc_tar_ssvm = test_svm(model_tar_ssvm, labels.test.target, data.test.target);
    accuracy_tar_ssvm(i) = acc_tar_ssvm(1);
    
    % A-SVM
    tstart = tic;
    model_asvm = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.ASVM, model_src);
    telapsed_asvm(i) = toc(tstart);
    acc_asvm = test_svm(model_asvm, labels.test.target, data.test.target);
    accuracy_asvm(i) = acc_asvm(1);
    % fprintf('   ASVM: Accuracy = %6.2f (Time = %6.2f)\n', accuracy_tar(i), telapsed_tar(i));
    
    % PMT-SVM
    tstart = tic;
    model_pmt_svm = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.PMT_SVM, model_src);
    telapsed_pmt_svm(i) = toc(tstart);
    acc_pmt_svm = test_svm(model_pmt_svm, labels.test.target, data.test.target);
    accuracy_pmt_svm(i) = acc_pmt_svm(1);
end
fprintf('\n');
fprintf('%s => %s: %s => %s\n', ...
    param.domain_abrv{source_domain}, param.domain_abrv{target_domain},...
 param.domain_names{source_domain}, param.domain_names{target_domain});
fprintf('       SRC:       Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_src), std(accuracy_src)/sqrt(n), mean(telapsed_src));
fprintf('       TAR:       Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_tar), std(accuracy_tar)/sqrt(n), mean(telapsed_tar));
fprintf('       SRC(SSVM): Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_src_ssvm), std(accuracy_src_ssvm)/sqrt(n), mean(telapsed_src_ssvm));
fprintf('       TAR(SSVM): Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_tar_ssvm), std(accuracy_tar_ssvm)/sqrt(n), mean(telapsed_tar_ssvm));
fprintf('       ASVM:      Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_asvm), std(accuracy_asvm)/sqrt(n), mean(telapsed_asvm));
fprintf('       PMT_SVM:   Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_pmt_svm), std(accuracy_pmt_svm)/sqrt(n), mean(telapsed_pmt_svm));