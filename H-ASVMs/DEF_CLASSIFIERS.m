classdef DEF_CLASSIFIERS
    % DEF_CLASSIFIERS
    %   classifier definitions
    
    properties (Constant)
        % SVMs, ASVMs
        SRC          = 1;
        TAR          = 2;
        MIX          = 3;
        ASVM         = 4;
        PMT_SVM      = 5;
        % Stuctrual SVMs
        SRC_SSVM      = 6;
        TAR_SSVM      = 7;
        TAR_ALL_SSVM  = 8;
        ASSVM         = 9;
        TAR_ALL_ASSVM = 10;
        HASVM         = 11;
        COSS          = 12;
    end

    methods (Access = private)    % private so that you cant instantiate
        function out = DEF_CLASSIFIERS
        end
    end
    
end

