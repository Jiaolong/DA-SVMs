function mid = get_model_id(target, target_domains, param)
% find the corresponding model id for the target domain
index = find(target == target_domains);
switch index
    case 1
        mid = param.DEF_MODEL_IDS.M_T1;
    case 2
        mid = param.DEF_MODEL_IDS.M_T2;
    case 3
        mid = param.DEF_MODEL_IDS.M_T3;
end
end