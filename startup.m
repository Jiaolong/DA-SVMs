addpath('./def/');
addpath('./util/');
addpath('./OfficeDemo/');
addpath('./SceneSegmentation/');

if ~exist('./cache', 'dir')
    mkdir('./cache');
end

% If use vl_feat
% dowload the code from: http://www.vlfeat.org/
%run('../external/vlfeat-0.9.17/toolbox/vl_setup.m');

% If Use mosek for optimization
% dowload the software from: http://www.mosek.com/resources/downloads