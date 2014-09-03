% test domain reshape
clear all;
startup;

addpath('domain_discovery');
addpath(genpath('domain_cluster_tree'));

param = config();
[Data, Labels] = load_data(param.DATA_DIR, param.norm_type);

a = 1; w = 2; d = 3; c = 4;
domains = [a w];

X = []; Y = []; D = [];
for i=1:length(domains)
    d    = domains(i);
    X1   = Data{d};
    Y1   = Labels{d};
    D1   = i*ones(length(Y1),1);
    X    = [X; X1];
    Y    = [Y; Y1'];
    D    = [D; D1];
end

k = 2; % number of domains to reshape
%-------Gong NIPS'13---------
fname = sprintf('./cache/reshape_k_%d_d_[aw]', k);
try
    load(fname);
catch
    z1 = domain_reshape( X, Y, k);
    [acc_1 z1] = ClusterAccuracy(z1, D);
    save(fname, 'z1', 'acc_1');
end

%-------Hoffman ECCV'12--------
z2 = DomainDiscovery (X, Y, k, D, param);
[acc_2 z2] = ClusterAccuracy(z2, D);

%------Clustering Tree---------
opts            = struct;
opts.depth      = 2;
opts.numTrees   = 100;
opts.numSplits  = 5;
opts.numDomains = k;
opts.verbose    = false;
opts.param      = param;
tree = treeTrain_domainCluster(X, [], opts, Y);
z3 = treeTest_domainCluster(tree, X);
[acc_3 z3] = ClusterAccuracy(z3, D);

% Visualization
im_gt = repmat(D, 1, 50);
im_z1 = repmat(z1, 1, 50);
im_z2 = repmat(z2, 1, 50);
im_z3 = repmat(z3, 1, 50);
subplot(1,4,1);
imagesc(im_gt);
title('Ground truth');
subplot(1,4,2);
imagesc(im_z1);
title(sprintf('NIPS Domain Reshape %f ', acc_1));
subplot(1,4,3);
imagesc(im_z2);
title(sprintf('ECCV Domain Discovery %f ', acc_2));
subplot(1,4,4);
imagesc(im_z3);
title(sprintf('Clustering tree %f ', acc_3));