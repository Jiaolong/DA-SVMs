function labels = update_labels(labels, param)
% UpdateLabelValues
% post-process step to make values work
if ~isfield(param, 'all_categories')
    return;
end
all_categories = param.all_categories;
categories = param.categories;
labels.train.source = UpdateLabels(labels.train.source, all_categories, ...
    categories);
labels.train.target = UpdateLabels(labels.train.target, all_categories, ...
    categories);
labels.test.target = UpdateLabels(labels.test.target, all_categories, ...
    categories);
end

function labels = UpdateLabels(labels, all_categories, categories)
for i = 1:numel(labels)
    oldL = labels(i);
    l = all_categories{oldL};
    newL = find(strcmp(l, categories));
    labels(i) = newL;
end
end