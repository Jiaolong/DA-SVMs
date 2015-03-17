function model = Train(labels, data, param, domain)

if ~isfield(param, 'C_s') || ~isfield(param, 'C_t')
    param.C_s = 1;
    param.C_t = 1;
end
if ~isfield(param, 'gamma')
    param.gamma = 10^(-4);
end

if domain == param.source
    param.train_classes = sort(unique(labels.source));
    data_svm = AugmentWithOnes(param,data.source);
    labels_svm = labels.source;
    param.weights = param.C_s * ones(length(labels.source), 1);
    param.svm.C = param.C_s;
else
    param.train_classes = sort(unique(labels.target));
    data_svm = AugmentWithOnes(param,data.target);
    labels_svm = labels.target;
    param.weights = param.C_t * ones(length(labels.target), 1);
    param.svm.C = param.C_t;
end

svm = train(param.weights, labels_svm', sparse(data_svm),...
    sprintf('-B %f -c %f -q', param.svm.biasMultiplier, param.svm.C)) ;

w = svm.w' ;
model.svmmodel = svm;
model.b = param.svm.biasMultiplier * w(end, :) ;
model.w = w(1:end-1, :);
end

function aug_data = AugmentWithOnes(param, data)
if param.svm.biasMultiplier == -1
    aug_data = [data, ones(size(data,1),1)];
else
    aug_data = data;
end
end