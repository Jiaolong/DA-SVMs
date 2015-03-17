classdef DEF_PATH
    % DEF_PATH
    %   Define path
    
    properties (Constant)
        MOSEK_PATH = '/home/jiaolong/Software/mosek/7/toolbox/r2009b/';
        VLFEAT_PATH = '../external/vlfeat-0.9.17/';
        LIBLINEAR_PATH = '../liblinear-mmdt/matlab/';
        DOMAIN_TRANFORM_ECCV10 = './external/DomainTransformsECCV10/';
        ASVM_LINEAR_PATH = '../external/ASVM_Linear/matlab/';
        ASVMS_PATH = '../external/A-SVMs/';
        LBFGS_PATH = '../external/minConf/';
        DATA_OFFICE_PATH = './data_office/';
        DATA_OFFICE_IMAGES_PATH = '/home/jiaolong/DataSets/Data_Office/';
        DATA_SCENE_PATH = './data_scene/';
    end

    methods (Access = private)    % private so that you cant instantiate
        function out = DEF_PATH
        end
    end
    
end

