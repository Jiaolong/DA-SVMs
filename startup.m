addpath('./def/');
addpath('./util/');
addpath('./OfficeDemo/');
addpath('./SceneSegmentation/');

exists_or_mkdir('./cache');

% If use vl_feat
% dowload the code from: http://www.vlfeat.org/
%run('../external/vlfeat-0.9.17/toolbox/vl_setup.m');

% If Use mosek for optimization
% dowload the software from: http://www.mosek.com/resources/downloads