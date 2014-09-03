function models = mt_hasvm_opt( models, param)
% Usage:
%   models = mt_hasvm_opt( models, pool_size, param)
% Input:
% Output:

pool_size = param.pool_size;
% Set the current model in the mt_hasvm_fv_cache
[blocks, lb] = set_models(models);
% Set up the objective function evaluation/gradient oracle
obj_func = @(x) mt_hasvm_fv_obj_func(x, 2*pool_size);
% Vectorize the model parameters and get the box constraints
w  = cat(1, blocks{:});
lb = cat(1, lb{:});
ub = inf*ones(size(lb));

% Prepare the training examples residing in the cache for optimization
mt_hasvm_fv_cache('ex_prepare');
% Optimize the objective function on the cache with L-BFGS
% th = tic;
w = minConf_TMP(obj_func, w, lb, ub, param.lbfgs.options);
% th = toc(th);
% fprintf('Parameters optimized in %.4f seconds\n', th);

%---------------------------------------------------------------------
% Update the model with the its parameters
base = 1;
for m=2:length(models) % start from 2, ignore the source model
    for i = 1:models{m}.numblocks
        models{m}.blocks(i).w = w(base:base+models{m}.blocks(i).dim-1);
        base = base + models{m}.blocks(i).dim;
    end
end
end

