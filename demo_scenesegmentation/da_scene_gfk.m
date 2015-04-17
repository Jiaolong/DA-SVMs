clear all;

% add dependencies
addpath('./def/');
addpath(genpath('./H-ASVMs/'));
addpath('../liblinear-mmdt/matlab/');
addpath(genpath('./external/'));

cache_dir = './cache/ASVM/';
exists_or_mkdir([cache_dir 'SRC/']);
exists_or_mkdir([cache_dir 'TAR/']);

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

% prepare data
Xs = data.train.source;
Ys = labels.train.source;
Ps = princomp(Xs);  % source subspace

Xt = data.train.target;
Yt = labels.train.target;
Pt = princomp(Xt);  % target subspace

%------------------------------------------------------------------------------
% GFK
% Ps = PLS(Xr, OneOfKEncoding(Yr), 3*d);
% PLS generally leads to better performance.
% A nice implementation is publicaly available at http://www.utd.edu/~herve/

G = GFK([Ps,null(Ps')], Pt(:,1:d));
[~, accy] = my_kernel_knn(G, Xs, Ys, Xt, Yt);
fprintf('\t\t%2.2f%%\n', accy*100);
