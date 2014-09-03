function [v, g] = mt_hasvm_fv_obj_func(w, num_threads)
% Get the object function value and gradient at w.
%   [v, g] = fv_obj_func(w, num_threads)
%
% Return values
%   v             Objective function value f(w)
%   g             Objective function gradient \nabla f(w)
%
% Arguments
%   w             Gradient and function evaluation point
%   num_threads   Number of worker threads to use for computing the gradient

[v, g] = mt_hasvm_fv_cache('gradient', w, num_threads);
