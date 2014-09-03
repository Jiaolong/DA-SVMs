% Author: Judy Hoffman (jhoffman@eecs.berkeley.edu)

% ref: Discovering Latent Domains for Multisource Domain Adaptation
% J. Hoffman, B. Kulis, T. Darrell, K. Saenko
% In Proceedings European Conference in Computer Vision (ECCV), Florence, Italy, 2012
function [pred_domain_labels, m, objective,acc] = DomainDiscovery (X, y, ...
    num_domains, domain_labels, param)
%{
Inputs:
    X - data matrix
    y - category label vector
    num_domains - number of domains to discover
    domain_labels - ground truth domain labels (if they exist)
    (optional) param - can pass in parameters or set them at the top of this file.

Outputs:
    pred_domain_labels - vector specifying predicted domain cluster membership
    means - matrix of means of the domain clusters
    objective - value of optimization objective after clustering

Usage:
    Given data and class labels, uses the information to form domain
    clusters. Uses a hierarchical clustering method with class constraints.
    %}
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%         Parameters          %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    default_maxIter = 5; % max number of iterations for algorithm to run
    default_c_g = 1; % loss weight on global obj
    default_c_l = 1; % loss weight on local obj
    if nargin < 5
        param.maxIter = default_maxIter;
        param.c_g = default_c_g;
        param.c_l = default_c_l;
    else
        if ~isfield(param, 'max_iter')
            param.maxIter = default_maxIter;
        end
        if ~isfield(param, 'c_g');
            param.c_g = default_c_g;
        end
        if ~isfield(param, 'c_l');
            param.c_l = default_c_l;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    maxIter = param.maxIter;
    c_g = param.c_g;
    c_l = param.c_l;
    % Initialize
    %   mu - local mean
    %   Z_l - local assignment
    %   Z_g - global assignment
    %   m - global mean
    % fprintf('    Initializing clusters\n');
    [mu, Z_g, m, mu_label] = Initialize(X, y, num_domains, param);
    
    % Note: mu_label(j) specifies what object class the local cluster j is in
    
    objective = zeros(maxIter - 1, 1);
    thresh = 0.001; % if objective changes less than this ratio - stop.
    acc = zeros(maxIter-1, 1);
    iter = 1;
    converged = false;
    while not(converged) && iter < maxIter
        if mod(iter,10) == 0
            fprintf('iter=%d/%d\n',iter, maxIter);
        end
        
        %%% Update local assignment %%%
        Z_l = UpdateLocalAssignment(X, mu, y, mu_label);
        
        %%% Update local means %%%
        mu = UpdateLocalMean(X, Z_l, Z_g, m, c_g, c_l);
        
        %%% Update global global assignment %%%
        Z_g = UpdateGlobalAssignment(mu, m, mu_label);
        
        %%% Update global means %%%
        m = UpdateGlobalMean(mu, Z_g);
        
        [~, ~, objective(iter)] = ComputeObjective(X, Z_l, mu, Z_g, m, c_g, c_l);
        
        pred_domain_labels = Unbinarize(Z_g, Z_l);
        if ~isempty(domain_labels)
            acc(iter) = ClusterAccuracy(pred_domain_labels, domain_labels);
            % fprintf('Objective=%6.2f Local %6.2f Global %6.2f - Acc=%6.2f\n',objective(iter),lo(iter), go(iter),acc(iter)*100');
            % fprintf('Objective=%6.2f - Acc=%6.2f\n',objective(iter), acc(iter)*100');
        end
        if iter > 1
            converged = ((objective(iter-1) - objective(iter)) / objective(iter-1) < thresh);
        end
        iter = iter + 1;
    end
    
end

function [mu, Z_g, m, mu_label] = Initialize(X, y, k, param)
num_runs = 10;
obj_min = inf;
if isfield(param, 'c_g') && isfield(param, 'c_l')
    c_g = param.c_g;
    c_l = param.c_l;
else
    c_g = 1;
    c_l = 1;
end

for i = 1:num_runs
    [Z_l_, mu_, mu_label_] = InitializeLocalClusters(X, y, k);
    [Z_g_, m_] = InitializeGlobalClusters(mu_, mu_label_, k);
    [lo, go, obj] = ComputeObjective(X, Z_l_, mu_, Z_g_, m_, c_g, c_l);
    if go < obj_min
        obj_min = go;
        mu_label = mu_label_;
        mu = mu_;
        Z_g = Z_g_;
        m = m_;
    end
end
end

function [Z_l, mu, mu_label] = InitializeLocalClusters(X, y, k)
% initialize local clusters using standard k-means for per class data.
classes = sort(unique(y));
num_classes = length(classes);
local_clusters = zeros(length(y), 1);
mu = zeros(num_classes * k, size(X, 2));
mu_label = zeros(num_classes * k, 1);

for c = 1:num_classes
    assert(sum(y == classes(c)) >= k, 'must have at least k points per class');
    
    cluster = 0;
    while (max(cluster) < k)
        [cluster, m] = kmeans(X(y == classes(c), :), k, 'EmptyAction', 'drop');
    end
    local_clusters(y == classes(c)) = k * (c - 1) + cluster;
    mu((k * (c - 1) + 1):k*c, :) = m;
    mu_label((k * (c-1) + 1):k*c) = c;
end
Z_l = Binarize(local_clusters);
end

function [Z_g, m] = InitializeGlobalClusters(mu, mu_label, k)
m = KmeansPlusPlus(mu, k, 1);
Z_g = UpdateGlobalAssignment(mu, m, mu_label);
end

function Z_l = UpdateLocalAssignment(X, mu, y, mu_label)
% update local assignments according to normal k-means object and store
% label vector output that says the class label of each local cluster.

%compute (X(i,:)-mu(j,:))^2
I = size(X,1);
J = size(mu,1); %J = num_domains*numClasses
Z_l = zeros(I,J);

for i = 1:size(X,1)
    % find j for each i
    c = y(i);
    mu_pts = find(mu_label == c);
    mu_c = mu(mu_pts, :);
    v = sum((repmat(X(i, :),length(mu_pts), 1) - mu_c).^2, 2);
    [~, best_pt] = min(v);
    j = mu_pts(best_pt);
    Z_l(i,j) = 1;
end
end

function mu = UpdateLocalMean(X, Z_l, Z_g, m, c_g, c_l)
J = size(Z_l,2); %number of local clusters
d = size(X,2); %num pts x size of data pt
mu = zeros(J,d);

for j = 1:J %update each cluster mean
    mu_j = c_g * Z_g(j,:) * m + c_l * Z_l(:,j)' * X;
    normalizer = c_g * sum(Z_g(j,:)) + c_l * sum(Z_l(:,j));
    mu(j,:) = mu_j / normalizer;
end

end

function Z_g = UpdateGlobalAssignment(mu, m, label)
% updates global assignments Z_g to be set to the single local cluster mu_j
% that minimizes cost while satisfying the objective that two clusters j_1
% and j_2 can not be assigned to the same m_k if label(j_1)=label(j_2)

J = size(mu,1);
K = size(m,1);

classes = unique(label);
C = length(classes);

% generate all feasible assignments per class
classBasedAssignments = perms(1:K);

% pick feasible assignment that has the lowest objective
bestObjective = inf(C,1);
bestClassAssignments = zeros(J,1);

for i = 1:size(classBasedAssignments,1)
    % global means are fixed to each class can be updated independently.
    % check if lowers objective of the
    assignment = classBasedAssignments(i,:);
    for c = 1:C
        %objective = classBasedObjective(mu(label==c,:),m(assignment,:));
        objective = norm(mu(label==c,:)-m(assignment,:));
        if objective < bestObjective(c)
            bestObjective(c) = objective;
            bestClassAssignments(label==c,:) = assignment;
        end
    end
end

Z_g = Binarize(bestClassAssignments);

end

function Z = Binarize(assignments)
J = size(assignments,1);
K = max(assignments);
Z = zeros(J,K);
for j = 1:J
    k = assignments(j);
    Z(j,k) =  1;
end
end

function pred_domain_labels = Unbinarize(Z_g, Z_l)
pred_domain_labels = zeros(size(Z_l,1),1);
for i = 1:size(Z_l,1)
    if sum(Z_l(i,:)) ~= 1
        fprintf('pt %d is not assigned locally correctly\n',i);
    end
    k = find(Z_g(Z_l(i,:) == 1, :)==1);
    pred_domain_labels(i) = k;
end
end

function m = UpdateGlobalMean(mu,Z_g)
K = size(Z_g, 2);
m = zeros(K,size(mu,2));
for k = 1:K
    m(k,:) = Z_g(:,k)'*mu / sum(Z_g(:,k));
end
end
