classdef DEF_MODEL_IDS_3L
    % DEF_MODEL_IDS_3L
    %   Model ids in the hierarchical structure
    
    properties (Constant)
        M_S0 = 0;
        
        M_S1 = 1;

        M_T1 = 2;
        M_N1 = 3;
        
        M_T2 = 4;
        M_T3 = 5;
    end

    methods (Access = public)    % private so that you cant instantiate
        function out = DEF_MODEL_IDS_3L
        end
    end
    
end

