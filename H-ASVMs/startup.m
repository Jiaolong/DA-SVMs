% Use vl_feat
% dowload the code from: http://www.vlfeat.org/
%run('../external/vlfeat-0.9.17/toolbox/vl_setup.m');

% Add dependencies
% Use mosek for optimization
% dowload the software from: http://www.mosek.com/resources/downloads
MOSEK_PATH = '/home/jiaolong/Software/mosek/7/toolbox/r2009b/';
LIBLINEAR_WEIGHTS_PATH = '../external/liblinear-weights-1.91/matlab/';
ASVM_LINEAR_PATH = '../external/ASVM_Linear/matlab/';
ASVMS_PATH = '../external/A-SVMs/';
LBFGS_PATH = '../external/minConf/';

% addpath(MOSEK_PATH);
addpath(LIBLINEAR_WEIGHTS_PATH);
addpath(ASVM_LINEAR_PATH);
addpath(ASVMS_PATH);
addpath(genpath(LBFGS_PATH));

addpath('./train/');
addpath('./test/');
addpath('./domain_discovery/');
addpath('./util/');
addpath(genpath('./mt_hasvm/'));
addpath(genpath('./bin/'));

fprintf('H-ASVMs demo ready.\n');

if ~exist('./cache', 'dir')
    mkdir('./cache');
end