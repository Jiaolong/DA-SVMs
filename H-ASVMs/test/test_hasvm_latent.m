function test_hasvm_latent

param = config();
[Data, Labels] = load_data(param.DATA_DIR, param.norm_type);

a = 1; w = 2; d = 3; c = 4;
source_domain  = w;
target_domains = [a d c];

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

% Load discovered domains
% See run_discover_domains.m
% dir_index = './data_save/latent_domains_nips/' ;
dir_index = './data_save/latent_domains_eccv/' ;
str_domains = cat(2, param.domain_abrv{target_domains});
load([dir_index 'domain_index/latent_domains_' str_domains '_pr.mat']);

% Change the order of the targets
t1 = 1; t2 = 2; t3 = 3;
lat_target_domains = [t3 t1 t2];

Data_T   = cell(n_td,1);
Labels_T = cell(n_td,1);
X = cat(1, Data{target_domains,1});
Y = cat(2, Labels{target_domains,1});
for i=1:length(lat_target_domains)
    Data_T{i}   = X(z==lat_target_domains(i),:);
    Labels_T{i} = Y(z==lat_target_domains(i));
end


fprintf('       Iteration: %d', n);
for i = 1:n
    fprintf('...%d', n-i);
    labels_pr = [];
    labels_gt = [];
    % for each testing domain
    for v=1:n_td
        test_domain    = lat_target_domains(v);
        test_domain_id = find(lat_target_domains == test_domain);
        
        % Load data splits
        train_ids             = load_splits(source_domain, target_domains(1), param);
        data.train.source     = Data{source_domain}(train_ids.source{i}, :);
        labels.train.source   = Labels{source_domain}(train_ids.source{i});
        
        data.train.targets           = cell(1,n_td);
        data.test.targets            = cell(1,n_td);
        labels.train.targets         = cell(1,n_td);
        data.train.targets_model_ids = cell(1,n_td);
        labels.test.targets          = cell(1,n_td);
        fname = [dir_index 'train_test_splits/latent_domains_SRC_' param.domain_abrv{source_domain} '_splits_r_' num2str(i) '_.mat'];
        train_ids = cell(n_td,1);
        test_ids  = cell(n_td,1);
        try
            load(fname);
        catch
            for t=1:n_td
                id = lat_target_domains(t);
                [train_ids{id}, test_ids{id}]   = gen_splits(Labels_T{id}, 3);
            end
            save(fname, 'train_ids', 'test_ids');
        end
        
        for t=1:n_td
            id = lat_target_domains(t);
            data.train.targets{t}   = Data_T{t}(train_ids{id}, :);
            data.test.targets{t}    = Data_T{t}(test_ids{id}, :);
            mid = get_model_id(id, lat_target_domains, param);
            data.train.targets_model_ids{t}  = mid*ones(1, size(data.train.targets{t},1));
            labels.train.targets{t} = Labels_T{t}(train_ids{id});
            labels.test.targets{t}  = Labels_T{t}(test_ids{id});
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
        telapsed_ssvm(i, test_domain) = toc(tstart);
        accuracy_ssvm(i, test_domain) = test_svm(model_src_ssvm, labels.test.targets{test_domain_id}, data.test.targets{test_domain_id}, param);
        
        % H-ASVMs
        tstart = tic;
        [~, model_hasvms] = train_ssvms(labels.train, data.train, param, DEF_CLASSIFIERS.HASVM, model_src_ssvm);
        telapsed_hasvm(i, test_domain) = toc(tstart);
        
        mid = get_model_id(test_domain, lat_target_domains, param);
        model_hasvm = get_hsvm_model(mid, model_hasvms);
        [accuracy_hasvm(i, test_domain), pl] = test_svm(model_hasvm, labels.test.targets{test_domain_id}, data.test.targets{test_domain_id}, param);
        labels_pr = [labels_pr pl];
        labels_gt = [labels_gt labels.test.targets{test_domain_id}];
    end
    avg_acc(i) = multiclass_acc( labels_pr, labels_gt, 10 );
end

for t = lat_target_domains
    fprintf('\n');
    fprintf('%s => %s:\n', ...
        param.domain_abrv{source_domain}, param.lat_domain_abrv{t});
    fprintf('       SRC(SSVM): Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
        mean(accuracy_ssvm(:,t)), std(accuracy_ssvm(:,t))/sqrt(n), mean(telapsed_ssvm(:,t)));
    fprintf('       HASVM:     Mean Accuracy = %6.3f +/- %6.2f  (Mean time = %6.3f)\n', ...
        mean(accuracy_hasvm(:,t)), std(accuracy_hasvm(:,t))/sqrt(n), mean(telapsed_hasvm(:,t)));
end
fprintf('\n');
fprintf('       Average Accuracy = %6.3f +/- %6.2f \n', ...
    mean(avg_acc(:)), std(avg_acc(:))/sqrt(n));
end

function [train_ids test_ids] = gen_splits(Y, train_per_class)
% Generate the random train/test splits
class = unique(Y);
Y = reshape(Y, length(Y), 1);
% assert(length(class) == 10, 'Categories are not complete!');
train_ids = [];
test_ids  = [];
for i=1:length(class)
    inds_c = find(Y == class(i));
    % assert(length(inds_c) > train_per_class, 'The category does not contain sufficient samples.');
    if length(inds_c) > train_per_class
        rand_ids = randperm(length(inds_c));
        train_ids_c = inds_c(rand_ids(1:train_per_class));
        test_ids_c  = inds_c(rand_ids(train_per_class+1:end));
        train_ids   = [train_ids; train_ids_c];
        test_ids    = [test_ids; test_ids_c];
    else
        test_ids    = [test_ids; inds_c];
    end
end
end
