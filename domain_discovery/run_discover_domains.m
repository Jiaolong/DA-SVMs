function run_discover_domains()
% discover all latent domains

clear all;
startup;
param = config();
[Data, Labels] = load_data(param.DATA_DIR, param.norm_type);

a = 1; w = 2; d = 3; c = 4;
use_gt =  true;
method = 'eccv'; % nips
%------------------------
source_domain  = c;
target_domains = [a w d];
% Predict the target labels
[X Y D] = pred_target_labels(param, source_domain, target_domains, Data, Labels, use_gt);
% Domain discovery
find_domains(param, target_domains, X, Y, D, method, use_gt);

%------------------------
source_domain  = d;
target_domains = [a w c];
% Predict the target labels
[X Y D] = pred_target_labels(param, source_domain, target_domains, Data, Labels, use_gt);
% Domain discovery
find_domains(param, target_domains, X, Y, D, method, use_gt);

%------------------------
source_domain  = a;
target_domains = [w d c];
% Predict the target labels
[X Y D] = pred_target_labels(param, source_domain, target_domains, Data, Labels, use_gt);
% Domain discovery
find_domains(param, target_domains, X, Y, D, method, use_gt);

%------------------------
source_domain  = w;
target_domains = [a d c];
% Predict the target labels
[X Y D] = pred_target_labels(param, source_domain, target_domains, Data, Labels, use_gt);
% Domain discovery
find_domains(param, target_domains, X, Y, D, method, use_gt);
end

function [X Y_pred D] = pred_target_labels(param, source_domain, target_domains, Data, Labels, use_gt)
if nargin < 6
    use_gt = false;
end
% ground truth domain labels
D = [];
for i=1:length(target_domains)
    d = target_domains(i);
    D = [D; i*ones(length(Labels{d}),1)];
end
X = cat(1, Data{target_domains});
Y = cat(2, Labels{target_domains});
if use_gt
    Y_pred = Y;
    return;
end
% Predeict target labels
data.source     = Data{source_domain,1};
labels.source   = Labels{source_domain,1};
% Train the source model
model = train_ssvms(labels, data, param, DEF_CLASSIFIERS.SRC_SSVM);
% Predict the target labels
[acc Y_pred] = test_svm(model, Y, X);
fprintf('\nSource classifier accuracy = %0.2f\n', acc);
end

function [z acc] = find_domains(param, target_domains, X, Y, D, method, use_gt)
% Find latent domains
% Store results:
n_td = length(target_domains); % number of target domains
str_domains = cat(2, param.domain_abrv{target_domains});
if use_gt
    fname = ['./cache/latent_domains_' str_domains '_.mat'];
else
    fname = ['./cache/latent_domains_' str_domains '_pr.mat'];
end

switch method
    case 'eccv'
        [z acc] = discover_domains_eccv(X, Y, n_td, D);
    case 'nips'
        [z acc] = discover_domains_nips(X, Y, n_td, D);
end
save(fname, 'z', 'acc');
end

function [z acc] = discover_domains_nips(X, Y, n_domains, D)
% Find latent domains
z = domain_reshape( X, Y, n_domains);
[acc z] = ClusterAccuracy(z, D);
% Visualization
im_gt = repmat(D, 1, 50);
im_z = repmat(z, 1, 50);
subplot(1,2,1);
imagesc(im_gt);
title('Ground truth');
subplot(1,2,2);
imagesc(im_z);
title(sprintf('NIPS Domain Reshape %f ', acc));
end

function [z acc] = discover_domains_eccv(X, Y, n_domains, D)
% Find latent domains
param = config();
%-------Hoffman ECCV'12--------
z = DomainDiscovery (X, Y, n_domains, D, param);
[acc z] = ClusterAccuracy(z, D);
% Visualization
im_gt = repmat(D, 1, 50);
im_z = repmat(z, 1, 50);
subplot(1,2,1);
imagesc(im_gt);
title('Ground truth');
subplot(1,2,2);
imagesc(im_z);
title(sprintf('ECCV Domain Discovery %f ', acc));
end
