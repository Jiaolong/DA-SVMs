function models = mt_hasvm_model_defines(model_s0, C, Layers, DEF_MODEL_IDS)
% Define the hierarchy
% Important note: change the corresponding definitions in DEF_MODEL_IDS
switch Layers
    case 2
        models = hierarchy_2layers(model_s0, C, DEF_MODEL_IDS);
    case 3
        models = hierarchy_3layers(model_s0, C, DEF_MODEL_IDS);
    otherwise
        error('Currently only support 2 and 3 layers adaptation tree!');
end
end

function models = hierarchy_2layers(model_s0, C, DEF_MODEL_IDS)
% Define application specific hierachical models
% Root
models{1} = model_s0; models{1}.id = DEF_MODEL_IDS.M_S0; models{1}.parent_id = DEF_MODEL_IDS.M_S0; models{1}.C = C; % Source model
% Layer 1
models{2} = model_s0; models{2}.id = DEF_MODEL_IDS.M_S1; models{2}.parent_id = DEF_MODEL_IDS.M_S0; models{2}.C = C;
% Layer 2
models{3} = model_s0; models{3}.id = DEF_MODEL_IDS.M_T1; models{3}.parent_id = DEF_MODEL_IDS.M_S1; models{3}.C = C;
models{4} = model_s0; models{4}.id = DEF_MODEL_IDS.M_T2; models{4}.parent_id = DEF_MODEL_IDS.M_S1; models{4}.C = C;
models{5} = model_s0; models{5}.id = DEF_MODEL_IDS.M_T3; models{5}.parent_id = DEF_MODEL_IDS.M_S1; models{5}.C = C;
end

function models = hierarchy_3layers(model_s0, C, DEF_MODEL_IDS)
% Define application specific hierachical models
% Root
models{1} = model_s0; models{1}.id = DEF_MODEL_IDS.M_S0; models{1}.parent_id = DEF_MODEL_IDS.M_S0; models{1}.C = C; % Source model
% Layer 1
models{2} = model_s0; models{2}.id = DEF_MODEL_IDS.M_S1; models{2}.parent_id = DEF_MODEL_IDS.M_S0; models{2}.C = C;
% Layer 2
models{3} = model_s0; models{3}.id = DEF_MODEL_IDS.M_T1; models{3}.parent_id = DEF_MODEL_IDS.M_S1; models{3}.C = C;
models{4} = model_s0; models{4}.id = DEF_MODEL_IDS.M_N1; models{4}.parent_id = DEF_MODEL_IDS.M_S1; models{4}.C = C;
% Layer 3
models{5} = model_s0; models{5}.id = DEF_MODEL_IDS.M_T2; models{5}.parent_id = DEF_MODEL_IDS.M_N1; models{5}.C = C;
models{6} = model_s0; models{6}.id = DEF_MODEL_IDS.M_T3; models{6}.parent_id = DEF_MODEL_IDS.M_N1; models{6}.C = C;
end