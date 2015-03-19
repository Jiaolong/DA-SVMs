function model = train_svms(labels_train, data_train, param, classifier_type, model_src)
% model = train_svms(labels_train, data_train, param, model_src, classifier_type)
% Train a multiclass classifier with SVMs
% Usage:
% Input:
% Output:

labels   = [];
data     = [];
num_classes  = length(unique(labels_train));
ws_zeros = zeros(size(data_train.target, 2), num_classes);

switch classifier_type
    case DEF_CLASSIFIERS.SRC
        labels = labels_train.source;
        data   = data_train.source;
        ws     = ws_zeros;
        param.weights = param.svm.C_s *ones(length(labels), 1);
        param.svm.C = param.svm.C_s;
    case DEF_CLASSIFIERS.TAR
        labels = labels_train.target;
        data   = data_train.target;
        ws     = ws_zeros;
        param.weights = param.svm.C_t*ones(length(labels), 1);
        param.svm.C = param.svm.C_t;
    case {DEF_CLASSIFIERS.ASVM, DEF_CLASSIFIERS.PMT_SVM}
        labels = labels_train.target;
        data   = data_train.target;
        ws     = model_src.w;
        param.weights = param.svm.C *ones(length(labels), 1);
end

svm_solver = param.svm.solver(classifier_type);
switch svm_solver
    case DEF_SVM_SOLVERS.PEGASOS % Pegasos
        lambda = 1 / (param.svm.C *  length(labels)) ;
        w = [] ;
        for ci = 1:num_classes
            % parfor ci = 1:num_classes
            % perm = randperm(size(data,2)) ;
            % fprintf('Training model for class %s\n', classes{ci}) ;
            y = 2 * (labels == ci) - 1 ;
            w(:,ci) = vl_pegasos(data', ...
                int8(y), lambda, ...
                'NumIterations', 50/lambda, ...
                'BiasMultiplier', param.svm.biasMultiplier) ;
        end
    case DEF_SVM_SOLVERS.LIBLINEAR % Liblinear
        svm = train(param.weights, ...
            labels', ...
            sparse(data),  ...
            sprintf('-B %f -c %f -q', ...
            param.svm.biasMultiplier, param.svm.C)) ;
        w = svm.w' ;
        model.svmmodel = svm;
		model.Label = svm.Label;
    case DEF_SVM_SOLVERS.ASVM_LINEAR % asvm_linear
        w = [];
        for ci = 1:num_classes
            if classifier_type ~= DEF_CLASSIFIERS.ASVM
                m_src = {};
            else
                svmsrc = model_src.svmmodel(ci);
                % svmsrc.w = svmsrc.w/norm(svmsrc.w(:)); % l2 normalization
                m_src = {svmsrc};
            end
            
            y = 2*(labels == ci) - 1;
            
            svm = asvmlinear_train(y', ...
                sparse(data),  ...
                m_src, ...
                1, ...
                sprintf('-B %f -c %f -q', ...
                param.svm.biasMultiplier, param.svm.C));
            
            w(:, ci) = svm.w';
            model.svmmodel(ci) = svm;
			model.Label = [1:num_classes];
        end
    case DEF_SVM_SOLVERS.MOSEK_QP % Mosek QP
        if classifier_type ~= DEF_CLASSIFIERS.PMT_SVM
            w = [];
            for ci = 1:num_classes
                y = 2 * (labels == ci) - 1 ;
                ws_ci = ws(:,ci);
                if norm(ws_ci(:))
                    ws_ci = ws_ci/norm(ws_ci(:));
                end
                svm = A_SVM(y', data, param.svm.C, ws_ci);
                w(:, ci) = [svm.w; svm.b] ;
            end
        else % PTM_SVM
            w = [];
            for ci = 1:num_classes
                y = 2 * (labels == ci) - 1 ;
                ws_ci = ws(:,ci);
                if norm(ws_ci(:))
                    ws_ci = ws_ci/norm(ws_ci(:));
                end
                svm = PMT_SVM(y', data, param.svm.C, ws_ci);
                w(:, ci) = [svm.w; svm.b] ;
            end
        end
end

model.b = param.svm.biasMultiplier * w(end, :) ;
model.w = w(1:end-1, :) ;
end
