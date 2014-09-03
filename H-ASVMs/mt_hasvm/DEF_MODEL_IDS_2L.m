classdef DEF_MODEL_IDS_2L
    % DEF_MODEL_IDS_2L
    %   Model ids in the hierarchical structure
    
    properties (Constant)
        M_S0 = 0;
        
        M_S1 = 1;

        M_T1 = 2;
        M_T2 = 3;
        M_T3 = 4;
    end

    methods (Access = public)    % private so that you cant instantiate
        function out = DEF_MODEL_IDS_2L
        end
    end
    
end

