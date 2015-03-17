function z = domain_reshape( X, Y, k)
% Domain reshape
% z = domain_reshape( X, Y, k )
% Input:
%        X  - features
%        Y  - labels
%        k  - number of domains
% Output:
%        z  - domain label
% Implementation of the algorithm in:
% "Reshaping Visual Datasets for Domain Adaptation. B. Gong, K. Grauman, and F. Sha.
% Proceedings of the Neural Information Processing Systems (NIPS), Lake Tahoe, NV, Dec. 2013."
% jiaolong@cvc.uab.es
% 2014-02-17


M = size(X, 1); %   Number of samples
K = X*X';

D2  = repmat(diag(K),1,size(K,1)) + repmat(diag(K)',size(K,1),1) - 2*K;
dm2 = median(D2(:));

% Kernel matrix
K = exp(-D2/dm2);

% Construct the matrix blocks as following:
% [beta_1; beta_2, ..., beta_k]'*P'*Q*P*[beta_1; beta_2, ..., beta_k]
% where
%       Q is a diagoal block matrices of K,
%           e.g. [K 0 0 ...; 0 K 0 ...; 0 0 K ...]
%       P*[beta_1; beta_2; ... beta_k]
%         = [beta_1' - beta_2'; beta_1 - beta_3';...,beta_{k-1} - beta_k]

w_block = size(K,1);
n_pairs = k*(k-1)/2; % number of domain pairs
n_rows  = n_pairs*w_block;
n_cols  = k*w_block;
P       = zeros(n_rows, n_cols);
Q       = zeros(n_rows);
I_k     = eye(w_block);

% Create Q
for i=1:n_pairs
    indx_colums = (i-1)*w_block + 1 : i*w_block;
    indx_rows   = indx_colums;
    Q(indx_colums, indx_rows) = K;
end
% Create P
s = 1;
for i=1:k-1
    for j=i+1:k
        indx_colums_pos = (i-1)*w_block + 1 : i*w_block;
        indx_colums_neg = (j-1)*w_block + 1 : j*w_block;
        indx_rows       = (s-1)*w_block + 1 : s*w_block;
        P(indx_rows, indx_colums_pos) = I_k;
        P(indx_rows, indx_colums_neg) = -I_k;
        s = s + 1;
    end
end
% New matrix
J = P'*Q*P;

H = -2*J;   f = zeros(M*k, 1);
deta = 0.01;
% Inequality 1: (\sum_m beta_{m,k}*y_{m,c}) <= (1+deta)(\sum_m y_{m,c})/M
A_k = OneOfKEncoding(Y)';
C   = size(A_k,1); % number of categories
A1  = zeros(k*C, M*k);
for i=1:k
    indx_colums = (i-1)*M+1:i*M;
    indx_rows   = (i-1)*C+1:i*C;
    A1(indx_rows, indx_colums) = A_k;
end
b_k = (1+deta)*sum(A_k,2)/M;
b1  = repmat(b_k, k, 1);
% Inequality 2: -(\sum_m beta_{m,k}*y_{m,c}) <= -(1-deta)(\sum_m y_{m,c})/M
A2  = -A1;
b_k = -(1-deta)*sum(A_k,2)/M;
b2  = repmat(b_k, k, 1);
% Inequality 3: \sum_k beta_{m,k} <= 1/C
I_Mk = repmat(eye(M), k, k);
A3 = I_Mk;                    b3 = ones(M*k, 1)/C;
% Inequality 4: -\sum_k beta_{m,k} <= -1/M
A4 = -I_Mk;                   b4 = -ones(M*k, 1)/M;
A = [A1;A2;A3;A4];            b  = [b1;b2;b3;b4];
% Equality: \sum_{m=1}^M beta_{m,k} = 1
Aeq = zeros(k,M*k);
for i=1:k
    indx_colums = (i-1)*M+1:i*M;
    Aeq(i, indx_colums) = ones(1, M);
end
beq = ones(k, 1);
lb = zeros(k*M,1);            ub = [];

% Call QP optimization
options = optimset('Display','final','Algorithm', 'active-set');
t1 = tic;
beta = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
t2 = toc(t1);
fprintf('\nQP takes %0.2f minutes. \n', t2/60);
% Compute the domain labels
beta   = reshape(beta, M, k);
[~, z] = max(beta, [], 2);
end

function Yk = OneOfKEncoding(Y)
% Class labels in Y are 1,2,3,...
Yk = full(sparse(1:length(Y), Y, ones(length(Y),1)));
end
