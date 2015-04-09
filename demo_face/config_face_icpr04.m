function param = config_face_icpr04(source, target)

frontal = 1; % frontal face, -30 ~ +30
l_45_60 = 2; % left face, 45 ~ 60
l_75_90 = 3; % left face, 75 ~ 90
r_45_60 = 4; % right face, 45 ~ 60
r_75_90 = 5; % tight face, 75 ~ 90

param.domains = [frontal, l_45_60, l_75_90, r_45_60, r_75_90];
param.domain_names = ...
    {'frontal', 'left_45_60', 'left_75_90', 'right_45_60', 'right_75_90'};
param.use_Gaussian_kernel = false;

param.num_class = 15;
param.num_tar_train = 5; % number of target samples per subject for DA training

% Directory containing the data 
param.data_dir = './data_face/ICPR04/';
param.cache_dir = './cache/Face/ICPR04/';
exists_or_mkdir(param.cache_dir);

% Choose the experiment type
param.held_out_categories = false; 

% Choose domains
if nargin == 2
    param.source = source;
    param.target = target;
else
    param.source = frontal;
    param.target = l_45_60;
end

% Choose the number of iterations to use
param.num_trials = 1;

% Choose dimension for data (with no dim reduction choose 225)
param.dim = 100;

% Choose the data normalization to use: ('none', 'l1','l2', 'l1_zscore',
% 'l2_zscore')
param.norm_type = 'l2_zscore';

% Parameters for MMDT
param.svm.C_s = .05;
param.svm.C_t = 1;
param.mmdt_iter = 2;

% Parameters for SVMs
param.svm.C = 1;
param.svm.biasMultiplier = 1;
param.svm.solver(DEF_CLASSIFIERS.SRC)     = DEF_SVM_SOLVERS.ASVM_LINEAR;
param.svm.solver(DEF_CLASSIFIERS.TAR)     = DEF_SVM_SOLVERS.ASVM_LINEAR;
param.svm.solver(DEF_CLASSIFIERS.ASVM)    = DEF_SVM_SOLVERS.ASVM_LINEAR;
param.svm.solver(DEF_CLASSIFIERS.PMT_SVM) = DEF_SVM_SOLVERS.MOSEK_QP;

% Parameters for SSVMs
param.ssvm.C = 0.001;
param.lbfgs.options.verbose = 0;
param.lbfgs.options.maxIter = 1000;
param.lbfgs.options.optTol  = 0.000001;

% Matlab pool size
param.pool_size = 4;

param.classifier_names = {
    'SRC (Liblinear)',...
    'TAR (Liblinear)',...
    'MIX',...
    'ASVM',...
    'PMT-SVM',...
    'SRC (SSVM)',...
    'TAR (SSVM)',...
    'TAR-ALL',...
    'ASSVM',...
    'TAR-ALL-ASSVM',...
    'HASVM',...
    'COSS'};

% Number of training examples per category
%param.num_train_source = 2000;
%param.num_train_target = 500;
end