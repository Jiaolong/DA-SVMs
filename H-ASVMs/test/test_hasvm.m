% Test H-ASVM
param = config();
[Data, Labels] = load_data(param.DATA_DIR, param.norm_type);

a = 1; w = 2; d = 3; c = 4;

source_domain  = a;
target_domains = [d w c]; % for 3 layers hierarchy, the order matters, i.e. [1, [2, 3]]

param = config(source_domain, target_domains(1));
% Hierarchy definition
param.DEF_MODEL_IDS = DEF_MODEL_IDS_3L;
param.model_define = @(m, C) mt_hasvm_model_defines(m, C, 3, param.DEF_MODEL_IDS);

% Store results:
n = param.num_trials;
n_td = length(target_domains); % number of target domains
telapsed_ssvm  = zeros(n, n_td);
accuracy_ssvm  = zeros(n, n_td);
telapsed_hasvm = zeros(n, n_td);
accuracy_hasvm = zeros(n, n_td);
avg_acc        = zeros(n, 1);

fprintf('       Iteration: %d', n);
for i = 1:n
    fprintf('...%d', n-i);
    labels_pr = [];
    labels_gt = [];
    % For each testing domain
    for v=1:n_td
        test_domain    = target_domains(v);
        test_domain_id = v;
        % Load data splits
        [train_ids, test_ids] = load_splits(source_domain, target_domains(1), param);
        data.train.source     = Data{source_domain}(train_ids.source{i}, :);
        labels.train.source   = Labels{source_domain}(train_ids.source{i});
        
        data.train.targets           = cell(1,n_td);
        data.test.targets            = cell(1,n_td);
        labels.train.targets         = cell(1,n_td);
        data.train.targets_model_ids = cell(1,n_td);
        labels.test.targets          = cell(1,n_td);
        for t=1:n_td
            [train_ids, test_ids]   = load_splits(source_domain, target_domains(t), param);
            data.train.targets{t}   = Data{target_domains(t)}(train_ids.target{i}, :);
            data.test.targets{t}    = Data{target_domains(t)}(test_ids.target{i}, :);
            mid = get_model_id(target_domains(t), target_domains, param);
            data.train.targets_model_ids{t}  = mid*ones(1,size(data.train.targets{t},1));
            labels.train.targets{t} = Labels{target_domains(t)}(train_ids.target{i});
            labels.test.targets{t}  = Labels{target_domains(t)}(test_ids.target{i});
        end
        
        if param.dim < size(data.train.source, 2)
            tr_targets = cat(1, data.train.targets{:});
            te_targets = data.test.targets{test_domain_id};
            P = princomp([data.train.source; ...
                tr_targets;...
                te_targets;]);
            
            data.train.source = data.train.source * P(:, 1:param.dim);
            
            for t=1:n_td
                data.train.targets{t} = data.train.targets{t} * P(:, 1:param.dim);
                data.test.targets{t}  = data.test.targets{t} * P(:, 1:param.dim);
            end
        end
        
        % SRC
        tstart = tic;
        model_src_ssvm = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
        telapsed_ssvm(i) = toc(tstart);
        
        accuracy_ssvm(i, test_domain_id) = test_svm(model_src_ssvm, labels.test.targets{test_domain_id}, data.test.targets{test_domain_id}, param);
        
        % H-ASVMs
        tstart = tic;
        [~, model_hasvms] = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.HASVM, model_src_ssvm);
        telapsed_hasvm(i, test_domain_id) = toc(tstart);
        
        mid = get_model_id(test_domain, target_domains, param);
        model_hasvm = get_hsvm_model(mid, model_hasvms);
        [accuracy_hasvm(i, test_domain_id), pl] = test_svm(model_hasvm, labels.test.targets{test_domain_id}, data.test.targets{test_domain_id}, param);
        
        labels_pr = [labels_pr pl];
        labels_gt = [labels_gt labels.test.targets{test_domain_id}];
    end
    avg_acc(i) = multiclass_acc( labels_pr, labels_gt, 10 );
end

for t = target_domains
    test_domain_id = find(t==target_domains);
    fprintf('\n');
    fprintf('%s => %s: %s => %s\n', ...
        param.domain_abrv{source_domain}, param.domain_abrv{t},...
        param.domain_names{source_domain}, param.domain_names{t});
    fprintf('       SRC(SSVM): Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
        mean(accuracy_ssvm(:, test_domain_id)), std(accuracy_ssvm(:, test_domain_id))/sqrt(n), mean(telapsed_ssvm(:, test_domain_id)));
    fprintf('       HASVM:     Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
        mean(accuracy_hasvm(:, test_domain_id)), std(accuracy_hasvm(:, test_domain_id))/sqrt(n), mean(telapsed_hasvm(:, test_domain_id)));
end

fprintf('\n');
fprintf('       Average Accuracy = %6.3f +/- %6.2f \n', ...
    mean(avg_acc(:)), std(avg_acc(:))/sqrt(n));
