function svm_models = mt_hasvm(param, labels, data, model_ids, model_svm_src)
% svm_models = mt_hasvm(param, labels, data, model_ids, model_svm_src)
% Usage:
% Input:
% Output:

num_class   = param.num_class;
len_feat    = size(data,2);
model_s0    = init_model_m(num_class, len_feat, model_svm_src);


% Construct hierachical models
C = param.ssvm.C;
models = param.model_define(model_s0, C);

% Initialize feature vector cache
max_num     = length(labels)*num_class;
max_dim     = len_feat*num_class;
max_nbls    = num_class;
mt_hasvm_fv_cache('init', max_num, max_dim, max_nbls, length(models));

% Set model parameters in the mt_hasvm_fv_cache
set_models(models);

% Write features into cache
mt_hasvm_fv_cache('ex_prepare');
for i=1:size(data,1)
    dataid = i;
    write_feat_multiclass(dataid, model_ids(i), labels(i), data(i,:), param);
end

% mt_hasvm_fv_cache('ex_prepare');
% mt_hasvm_fv_cache('print');
% info = info_to_struct(mt_hasvm_fv_cache('info'));
% [num_entries, num_examples] = info_stats(info);
% Optimize with LBFGS
models = mt_hasvm_opt(models, param);

% Output models
svm_models = cell(length(models), 1);
for i=1:length(models)
    svm_models{i} = get_svm_model(models{i}, param);
end
% Free cache
mt_hasvm_fv_cache('free');
end

function model_svm = get_svm_model(model, param)
% Convert model to svm model
num_class   = param.num_class;
for i=1:num_class
    model_svm.w(:,i) = model.blocks(i).w;
end
model_svm.b = zeros(1, num_class);
model_svm.id = model.id;
model_svm.Label = 1:num_class;
end

function write_feat_multiclass(dataid, model_id, y, x, param)
% write a feature vector into cache for multiclass SSVM
num_class = param.num_class;
len_x     = length(x);
fvc_zeros = zeros(num_class*len_x, 1);
is_mined  = false;
bls       = (0:num_class-1)';
key       = [dataid; 0; y; 0; model_id]; % add a model id
for i=1:num_class
    psi = fvc_zeros;
    psi((i-1)*len_x + 1:i*len_x) = x;
    if y == i % belief
        is_belief = true;
        loss = 0;
    else
        is_belief = false;
        loss = 1;
    end
    mt_hasvm_fv_cache('add', int32(key), int32(bls), single(psi), ...
        int32(is_belief), int32(is_mined), loss);
end
end

function model = init_model_m(num_class, len_feat, model_svm_src)
model.numblocks              = num_class;
w = model_svm_src.w;
for i=1:num_class
    model.blocks(i).w        = w(:,i);
    model.blocks(i).dim      = len_feat;
    model.blocks(i).lb       = -Inf*ones(len_feat,1);
    model.blocks(i).reg_mult = 1;
    model.blocks(i).learn    = 1;
end
end
