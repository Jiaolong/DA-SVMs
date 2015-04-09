% Hierarchical Adaptive SVMs for Domain Adaptation
% Author: Jiaolong Xu
% E-mail: jiaolong@cvc.uab.es

clear all;

param = config_office();
[Data, Labels] = load_data_office(param.DATA_DIR, param.norm_type);

% classifiers = {DEF_CLASSIFIERS.ASVM, DEF_CLASSIFIERS.PMT_SVM,...
%     DEF_CLASSIFIERS.SRC_SSVM, DEF_CLASSIFIERS.TAR_SSVM,...
%     DEF_CLASSIFIERS.MIX, DEF_CLASSIFIERS.COCS...
%     DEF_CLASSIFIERS.TAR_ALL_SSVM, DEF_CLASSIFIERS.ASSVM,...
%     DEF_CLASSIFIERS.TAR_ALL_ASSVM, DEF_CLASSIFIERS.HASVM};
classifiers = {DEF_CLASSIFIERS.HASVM};

a = 1; w = 2; d = 3; c = 4;
source_domain  = d;
target_domains = [c a w];
target_domain  = target_domains(2);

param = config(source_domain, target_domain);
param.target_domains = target_domains;

for i=1:length(classifiers)
    % Testing classifier, see DEF_CLASSIFIERS.m
    da_classifier = classifiers{i};
    
    fprintf('%s => %s: %s => %s\n', ...
        param.domain_abrv{source_domain}, param.domain_abrv{target_domain},...
        param.domain_names{source_domain}, param.domain_names{target_domain});
    
    [accuracy telapsed] = domain_adaptation(param, Data, Labels, da_classifier);
end

fprintf('\n\n');

ma = mean(accuracy);
err = std(accuracy)/sqrt(numel(accuracy));
fprintf('%s -> %s: %6.2f +/-%6.1f\n',param.domain_abrv{source_domain}, ...
    param.domain_abrv{target_domain}, ma, err);

