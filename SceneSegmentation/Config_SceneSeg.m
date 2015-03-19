function param = Config_SceneSeg(source, target)

%%%%%               FIXED PARAMETERS FOR OFFICE DATASET             %%%%%%
virtual = 1; kitti = 2; cambi = 3;
param.domains = [virtual, kitti, cambi];
param.domain_names = {'virtual', 'kitti', 'camvid'};
param.use_Gaussian_kernel = false;

% <region id="-1" name="void" color="0 0 0"/>
% <region id="0" name="building" color="128 0 0"/>
% <region id="1" name="grass" color="0 128 0"/>
% <region id="2" name="tree" color="128 128 0"/>
% <region id="3" name="cow" color="0 0 128"/>
% <region id="4" name="sheep" color="0 128 128"/>
% <region id="5" name="sky" color="128 128 128"/>
% <region id="6" name="airplane" color="192 0 0"/>
% <region id="7" name="water" color="64 128 0"/>
% <region id="8" name="face" color="192 128 0"/>
% <region id="9" name="car" color="64 0 128"/>
% <region id="10" name="bicycle" color="192 0 128"/>
% <region id="11" name="flower" color="64 128 128"/>
% <region id="12" name="sign" color="192 128 128"/>
% <region id="13" name="bird" color="0 64 0"/>
% <region id="14" name="book" color="128 64 0"/>
% <region id="15" name="chair" color="0 192 0"/>
% <region id="16" name="road" color="128 64 128"/>
% <region id="17" name="cat" color="0 192 128"/>
% <region id="18" name="dog" color="128 192 128"/>
% <region id="19" name="body" color="64 64 0"/>
% <region id="20" name="boat" color="192 64 0"/>
       
% param.categories = {'building' 'grass'  'tree' ...
%     'cow' 'sheep'  'sky' 'airplane'  'water' ...
%     'face' 'car' 'bicycle' 'flower' 'sign' 'bird' ...
%     'book' 'chair' 'road' 'cat' 'dog' 'body' 'boat'};

% param.categories = {'building' 'grass'  'tree' ...
%     'cow' 'sheep'  'sky' 'airplane'  'water' ...
%     'face' 'car' 'bicycle'};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%                      PARAMETERS TO EDIT                       %%%%%%
% Directory containing the data 
param.data_dir_virtual = './data_scene/LauraVid/';
param.data_dir_real = './data_scene/CamVid/';

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