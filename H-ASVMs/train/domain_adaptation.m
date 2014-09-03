function [accuracy telapsed] = domain_adaptation(param, Data, Labels, da_classifier)
% [accuracy telapsed] = domain_adaptation(param, Data, Labels, da_classifier)
% Usage:
% Input:
% Output:

source_domain  = param.source;
target_domain  = param.target;
target_domains = param.target_domains;

n              = param.num_trials;
n_td           = length(target_domains); % number of target domains
accuracy       = zeros(n,1);
telapsed       = zeros(n,1);

fprintf('Iteration: %d', n);
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
    
    data.train.targets           = cell(1,n_td);
    data.test.targets            = cell(1,n_td);
    labels.train.targets         = cell(1,n_td);
    data.train.targets_model_ids = cell(1,n_td);
    labels.test.targets          = cell(1,n_td);
    
    if da_classifier == DEF_CLASSIFIERS.HASVM || DEF_CLASSIFIERS.TAR_ALL_ASSVM...
            || DEF_CLASSIFIERS.TAR_ALL_SSVM
        for t=1:n_td
            [train_ids, test_ids] = load_splits(source_domain, target_domains(t), param);
            data.train.targets{t}   = Data{target_domains(t)}(train_ids.target{i}, :);
            data.test.targets{t}    = Data{target_domains(t)}(test_ids.target{i}, :);
            mid = get_model_id(target_domains(t), target_domains);
            data.train.targets_model_ids{t}  = mid*ones(1,size(data.train.targets{t},1));
            labels.train.targets{t} = Labels{target_domains(t)}(train_ids.target{i});
            labels.test.targets{t}  = Labels{target_domains(t)}(test_ids.target{i});
        end
    end
    
    if param.dim < size(data.train.source, 2)
        if da_classifier == DEF_CLASSIFIERS.HASVM || DEF_CLASSIFIERS.TAR_ALL_ASSVM...
            || DEF_CLASSIFIERS.TAR_ALL_SSVM
            tr_targets = cat(1, data.train.targets{:});
            te_targets = cat(1, data.test.targets{:});
%             index = target_domain == target_domains;
%             te_targets = data.test.targets{index};
            P = princomp([data.train.source; ...
                tr_targets;...
                te_targets;]);
            for t=1:n_td
                data.train.targets{t} = data.train.targets{t} * P(:, 1:param.dim);
                data.test.targets{t}  = data.test.targets{t} * P(:, 1:param.dim);
            end
        else
            P = princomp([data.train.source; data.train.target; data.test.target]);
            data.train.target = data.train.target * P(:, 1:param.dim);
            data.test.target = data.test.target * P(:, 1:param.dim);
        end
        data.train.source = data.train.source * P(:, 1:param.dim);
    end
    
    % Train and test DA models
    [acc telps] = train_and_test(labels, data, param, da_classifier);
    accuracy(i) = acc;
    telapsed(i) = telps;
end
fprintf('\n');
fprintf('%s: Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
    param.classifier_names{da_classifier}, ...
    mean(accuracy), std(accuracy)/sqrt(n), mean(telapsed));
end

function [accuracy telapsed] = train_and_test(labels, data, param, method)
% [accuracy telapsed] = train_and_test(labels, data, param, method)
% Usage:
% Input:
% Output:

% Train
switch method
    case DEF_CLASSIFIERS.SRC
        % SRC (SVM)
        tstart = tic;
        model = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.TAR
        % TAR (SVM)
        tstart = tic;
        model = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.ASVM
        % A-SVM
        tstart = tic;
        model_src = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC);
        model = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.ASVM, model_src);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.PMT_SVM
        % PMT-SVM
        model_src = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC);
        tstart = tic;
        model = train_svms(labels.train, data.train, param, DEF_CLASSIFIERS.PMT_SVM, model_src);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.SRC_SSVM
        % SRC (SSVM)
        tstart = tic;
        model = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.TAR_SSVM
        % TAR (SSVM)
        tstart = tic;
        model = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR_SSVM);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.TAR_ALL_SSVM
        % TAR_ALL (SSVM)
        tstart = tic;
        model = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR_ALL_SSVM);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.ASSVM
        % ASSVM
        model_src = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
        tstart = tic;
        model = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.ASSVM, model_src);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.TAR_ALL_ASSVM
        % ASSVM
        model_src = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
        tstart = tic;
        model = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.TAR_ALL_ASSVM, model_src);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.HASVM
        % H-ASVMs
        model_src = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.SRC_SSVM);
        tstart = tic;
        [~, model_hasvms] = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.HASVM, model_src);
        telapsed = toc(tstart);
        mid = get_model_id(param.target, param.target_domains);
        model = get_hsvm_model(mid, model_hasvms);
    case DEF_CLASSIFIERS.MIX
        lbs = [labels.train.source labels.train.target];
        dt  = [data.train.source;data.train.target];
        tstart = tic;
        model  = mt_hasvm_single(param, lbs, dt, []);
        telapsed = toc(tstart);
    case DEF_CLASSIFIERS.COCS
        lbs = [labels.train.source labels.train.target];
        dt  = [data.train.source; data.train.target];
        domain_ids = [zeros(length(labels.train.source),1);ones(length(labels.train.target),1)];
        tstart = tic;
        model = mt_hasvm_cocs(param, lbs, dt, domain_ids);
        telapsed = toc(tstart);
end

% Test
if method == DEF_CLASSIFIERS.HASVM || DEF_CLASSIFIERS.TAR_ALL_ASSVM...
            || DEF_CLASSIFIERS.TAR_ALL_SSVM
    index = param.target == param.target_domains;
    accuracy = test_svm(model, labels.test.targets{index}, data.test.targets{index}, param);
else
    accuracy = test_svm(model, labels.test.target, data.test.target, param);
end
end