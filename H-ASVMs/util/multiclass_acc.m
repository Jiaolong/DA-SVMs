function acc = multiclass_acc( labels_pr, labels_gt, n_class )
% Compute average precision

numTest = zeros(n_class, 1);
numTest = binsum(numTest, ones(length(labels_gt),1), labels_gt);

% Compute the confusion matrix
idx = sub2ind([n_class, n_class], ...
    labels_gt, labels_pr) ;
confus = zeros(n_class) ;
confus = binsum(confus, ones(size(idx)), idx) ;

acc = 100*mean(diag(confus)./numTest);
end

