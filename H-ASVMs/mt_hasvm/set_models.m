function [blocks, lb] = set_models(models)
% Set models
blocks = [];
lb     = [];
for i=1:length(models)
    [blks_i, lb_i, rm_i, lm_i, cmps_i] = fv_model_args(models{i});
    if i> 1
        blocks = [blocks; blks_i];
        lb     = [lb; lb_i];
    end
    mt_hasvm_fv_cache('set_model', blks_i, lb_i, rm_i, lm_i, {}, models{i}.C, int32(models{i}.id), int32(models{i}.parent_id));
end
end

function [blocks, lb, reg_mult, learn_mult, comps] = fv_model_args(model)
% Get model parameters
blocks        = {model.blocks(:).w}';
lb            = {model.blocks(:).lb}';
reg_mult      = single([model.blocks(:).reg_mult]');
learn_mult    = single([model.blocks(:).learn]');
comps         = cell(1,1);
comps{1}      = (0:length(model.blocks)-1)';
end