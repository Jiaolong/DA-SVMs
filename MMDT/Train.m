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
    data_svm = AugmentWithOnes(data.source);
    labels_svm = labels.source;
    param.weights = param.C_s * ones(length(labels.source), 1);
else
    param.train_classes = sort(unique(labels.target));
    data_svm = AugmentWithOnes(data.target);
    labels_svm = labels.target;
    param.weights = param.C_t * ones(length(labels.target), 1);
end
model = train(param.weights, labels_svm', sparse(data_svm), '-c 1 -q');
end

function aug_data = AugmentWithOnes(data)
aug_data = [data, ones(size(data,1),1)];
end