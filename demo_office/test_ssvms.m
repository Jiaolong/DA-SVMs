% configuration
amazon = 1; webcam = 2; dslr = 3; caltech = 4;
source_domain  = 4; 
target_domain  = 2;
param = config_office(source_domain, target_domain);

% load data
[Data, Labels] = load_data_office(param.DATA_DIR, param.norm_type);

% Hierarchy definition, for SSVM
param.DEF_MODEL_IDS = DEF_MODEL_IDS_2L;
param.model_define = @(m, C) mt_hasvm_model_defines(m, C, 2, param.DEF_MODEL_IDS);

% Store results:
n = param.num_trials;
telapsed_ssvm  = zeros(n,1);
accuracy_ssvm  = zeros(n,1);
telapsed_assvm = zeros(n,1);
accuracy_assvm = zeros(n,1);
telapsed_mix   = zeros(n,1);
accuracy_mix   = zeros(n,1);
telapsed_cocs  = zeros(n,1);
accuracy_cocs  = zeros(n,1);

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
        data.train.source  = data.train.source * P(:, 1:param.dim);
        data.train.target  = data.train.target * P(:, 1:param.dim);
        data.test.target   = data.test.target * P(:, 1:param.dim);
    end
    
    % SRC
    tstart = tic;
    model_src_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
    telapsed_ssvm(i) = toc(tstart);
    acc_ssvm = test_svm(model_src_ssvm, labels.test.target, data.test.target);
    accuracy_ssvm(i) = acc_ssvm(1);
        
    % ASSVM
    tstart = tic;
    model_assvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.ASSVM, model_src_ssvm);
    telapsed_assvm(i) = toc(tstart);
    acc_assvm = test_svm(model_assvm, labels.test.target, data.test.target);
    accuracy_assvm(i) = acc_assvm(1);
    
    % MIX
    tstart = tic;
    model_mix = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.MIX, model_src_ssvm);
    telapsed_mix(i) = toc(tstart);
    acc_mix = test_svm(model_mix, labels.test.target, data.test.target);
    accuracy_mix(i) = acc_mix(1);
    
    % Cost-Sensitive SSVM
    tstart = tic;
    model_cocs = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.COSS, model_src_ssvm);
    telapsed_cocs(i) = toc(tstart);
    acc_cocs = test_svm(model_cocs, labels.test.target, data.test.target);
    accuracy_cocs(i) = acc_cocs(1);
end
fprintf('\n');
fprintf('%s => %s: %s => %s\n', ...
    param.domain_abrv{source_domain}, param.domain_abrv{target_domain},...
    param.domain_names{source_domain}, param.domain_names{target_domain});
fprintf('       SRC(SSVM): Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_ssvm), std(accuracy_ssvm)/sqrt(n), mean(telapsed_ssvm));
fprintf('       ASSVM:     Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_assvm), std(accuracy_assvm)/sqrt(n), mean(telapsed_assvm));
fprintf('       MIX:     Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_mix), std(accuracy_mix)/sqrt(n), mean(telapsed_mix));
fprintf('       COSS:     Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    mean(accuracy_cocs), std(accuracy_cocs)/sqrt(n), mean(telapsed_cocs));