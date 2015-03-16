classdef DEF_SVM_SOLVERS
    % DEF_SVM_SOLVERS
    %   Define various SVM solvers
    
    properties (Constant)
        PEGASOS      = 1; % Pegasos
        LIBLINEAR    = 2; % Liblinear
        MOSEK_QP     = 3; % Mosek QP
        ASVM_LINEAR  = 4; % Linear adaptive SVM
    end

    methods (Access = private)    % private so that you cant instantiate
        function out = DEF_SVM_SOLVERS
        end
    end
    
end

