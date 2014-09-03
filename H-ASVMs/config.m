function param = config(source, target)

%===============================================================
% FIXED PARAMETERS FOR OFFICE DATASET
%===============================================================
%{{
amazon = 1; webcam = 2; dslr = 3; caltech = 4;
param.domains = [amazon, webcam, dslr, caltech];
param.domain_names = {'amazon', 'webcam', 'dslr', 'caltech'};
param.domain_abrv  = {'A', 'W', 'D', 'C'};
param.lat_domain_abrv = {'T1', 'T2', 'T3'};
param.use_Gaussian_kernel = false;

param.categories = {'back_pack' 'bike'  'calculator' ...
    'headphones' 'keyboard'  'laptop_computer' 'monitor'  'mouse' ...
    'mug' 'projector' };
%}}

%===============================================================
% PARAMETERS TO EDIT
%===============================================================
%{{
% Directory containing the data
param.DATA_DIR = '../data/';
param.IMAGE_DIR = '/home/jiaolong/DataSets/Data_Office/';

% Choose the experiment type
param.held_out_categories = false;

% Choose domains
if nargin == 2
    param.source = source;
    param.target = target;
else
    param.source = amazon;
    param.target = dslr;
end

% Choose the number of iterations to use
param.num_trials = 20;
% Choose dimension for data (with no dim reduction choose 800)
param.dim = 20;
% Choose the data normalization to use: ('none', 'l1','l2', 'l1_zscore',
% 'l2_zscore')
param.norm_type = 'l2_zscore';

%-------------------------------------------------------------------------
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

% paramters for cost-sensitive SSVM
param.gamma = 0.9;

%-------------------------------------------------------------------------
% Number of training examples per category (Below is parameters from paper)
if param.source == amazon
    param.num_train_source = 20; % Use 20 for amazon and 8 for every other domain
else
    param.num_train_source = 8;
end
param.num_train_target = 3;
end
%}}