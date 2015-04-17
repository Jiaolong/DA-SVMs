function param = Config_SceneSeg(source, target)

%%%%%               FIXED PARAMETERS FOR OFFICE DATASET             %%%%%%
virtual = 1; kitti = 2; cambi = 3;
param.domains = [virtual, kitti, cambi];
param.domain_names = {'virtual', 'kitti', 'camvid'};
param.use_Gaussian_kernel = false;

param.num_class = 6;
param.data_dir_virtual = './data_scene/CamVid_plus_LauraVid_6class/';
param.data_dir_real = param.data_dir_virtual;
param.train_src_list = [param.data_dir_virtual '/trainSRC633.txt'];
param.train_tar_list = [param.data_dir_real '/trainTRG20_2.txt'];
param.test_tar_list = [param.data_dir_real '/evalList.txt'];
% Choose the experiment type
param.held_out_categories = false; 

% Choose domains
if nargin == 2
    param.source = source;
    param.target = target;
else
    param.source = virtual;
    param.target = kitti;
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