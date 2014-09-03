function model_svm = mt_hasvm_single(param, labels, data, model_svm_src)
% Usage:
% Input:
% Output:

% Construct hierachical models
C = param.ssvm.C;
% Initialize an empty parent model
num_class   = length(param.categories);
len_feat    = size(data,2);
if isempty(model_svm_src) % training a new model
    model_s0    = init_model_w(num_class, len_feat, zeros(len_feat, 1));
    model_s1    = init_model_w(num_class, len_feat, ones(len_feat, 1));
else % adapted from model_svm_src
    model_s0    = init_model_m(num_class, len_feat, model_svm_src);
    model_s1    = model_s0;
end
models{1} = model_s0; models{1}.id = param.DEF_MODEL_IDS.M_S0; models{1}.parent_id = param.DEF_MODEL_IDS.M_S0; models{1}.C = C;
models{2} = model_s1; models{2}.id = param.DEF_MODEL_IDS.M_S1; models{2}.parent_id = param.DEF_MODEL_IDS.M_S0; models{2}.C = C;

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
    model_id = param.DEF_MODEL_IDS.M_S1;
    write_feat_multiclass(dataid, model_id, labels(i), data(i,:), param);
end

% Optimize with LBFGS
models = mt_hasvm_opt(models, param);

% Output models
model = models{param.DEF_MODEL_IDS.M_S1+1};
model_svm.w = zeros(len_feat, num_class);
model_svm.b = zeros(1, num_class);
for i=1:num_class
    model_svm.w(:,i) = model.blocks(i).w;
end
% Free cache
mt_hasvm_fv_cache('free');
end

function write_feat_multiclass(dataid, model_id, y, x, param)
% write a feature vector into cache for multiclass SSVM
num_class = length(param.categories);
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

function model = init_model_w(num_class, len_feat, w)
model.numblocks              = num_class;
for i=1:num_class
    model.blocks(i).w        = w;
    model.blocks(i).dim      = len_feat;
    model.blocks(i).lb       = -Inf*ones(len_feat,1);
    model.blocks(i).reg_mult = 1;
    model.blocks(i).learn    = 1;
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
