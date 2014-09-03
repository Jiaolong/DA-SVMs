function [model_svm model_svms] = train_ssvms(labels_train, data_train, param, classifier_type, model_src)
% [model_svm model_svms] = train_ssvms(labels_train, data_train, param, classifier_type, model_src)
% Train a multiclass classifier with structural SVMs
% Usage:
% Input:
% Output:

model_svms = [];
switch classifier_type
    case DEF_CLASSIFIERS.SRC_SSVM
        labels = labels_train.source;
        data   = data_train.source;
        % Train a multiclass classifier with structural SVM
        model_svm  = mt_hasvm_single(param, labels, data, []);
    case DEF_CLASSIFIERS.TAR_SSVM
        labels = labels_train.target;
        data   = data_train.target;
        % Train a multiclass classifier with structural SVM
        model_svm  = mt_hasvm_single(param, labels, data, []);
    case DEF_CLASSIFIERS.MIX
        labels = [labels_train.source labels_train.target];
        data   = [data_train.source;data_train.target];
        % Train a multiclass classifier with structural SVM
        model_svm  = mt_hasvm_single(param, labels, data, []);
    case DEF_CLASSIFIERS.COSS
        labels = [labels_train.source labels_train.target];
        data   = [data_train.source; data_train.target];
        domain_ids = [zeros(length(labels_train.source),1);ones(length(labels_train.target),1)];
        % Train a multiclass classifier with structural SVM
        model_svm  = mt_hasvm_coss(param, labels, data, domain_ids);
    case DEF_CLASSIFIERS.TAR_ALL_SSVM
        labels = cat(2, labels_train.targets{:});
        data   = cat(1, data_train.targets{:});
        % Train a multiclass classifier with structural SVM
        model_svm  = mt_hasvm_single(param, labels, data, []);
    case DEF_CLASSIFIERS.ASSVM
        labels = labels_train.target;
        data   = data_train.target;
        % Train a multiclass classifier with adaptive structural SVM
        model_svm  = mt_hasvm_single(param, labels, data, model_src);
    case DEF_CLASSIFIERS.TAR_ALL_ASSVM
        labels = cat(2, labels_train.targets{:});
        data   = cat(1, data_train.targets{:});
        % Train a multiclass classifier with structural SVM
        model_svm  = mt_hasvm_single(param, labels, data, model_src);
    case DEF_CLASSIFIERS.HASVM
        labels = cat(2, labels_train.targets{:});
        data   = cat(1, data_train.targets{:});
        model_ids = cat(2, data_train.targets_model_ids{:});
        % Train a multiclass classifier with hierarchical
        % adaptive structural SVMs
        model_svms = mt_hasvm(param, labels, data, model_ids, model_src);
        model_svm  = model_svms{param.DEF_MODEL_IDS.M_T1}; % The first target domain
end
end