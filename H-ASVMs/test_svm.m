function [acc, pl, scores, confus, acc_overall] = test_svm(model, labels, data)
% Usage:
%   [acc pl] = test_svm(model, labels, data)
% Input
%   model  -trained model
%   labels -ground truth label
%   data   -test data, num_samplesxdim
% Output:
%   acc    -accuracy
%   pl     -predicted label
%   score  - score of each sample
%   confus - confusion maxtrix
%   acc_overall - overall accuracy = total_correct/ total_sample

data = data';
num_classes = size(model.w,2);
assert(num_classes == length(unique(labels)),...
    sprintf('The number of testing classes (%d)is not equal to the number of training classes (%d)',...
    length(unique(labels)), num_classes));
numTest = zeros(num_classes, 1);
numTest = binsum(numTest, ones(length(labels),1), labels);
numTest(numTest == 0) = Inf;

% Adjust the order of the weights
model = order_model(model);

% Estimate the class of the test images
scores = model.w' * data + model.b' * ones(1,size(data,2));

[~, imageEstClass] = max(scores, [], 1);

labels = reshape(labels, size(imageEstClass));
% Compute the confusion matrix
idx = sub2ind([num_classes, num_classes], labels, imageEstClass);
confus = zeros(num_classes);
confus = binsum(confus, ones(size(idx)), idx);

% Plots
% figure(1) ; clf;
% subplot(1,2,1) ;
% imagesc(scores) ; title('Scores') ;
% set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
% subplot(1,2,2) ;
% imagesc(confus) ;
% title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
%     100 * mean(diag(confus)./numTest) )) ;
% print('-depsc2', [conf.resultPath '.ps']) ;
acc = 100*mean(diag(confus)./numTest);
pl = imageEstClass;

correct = sum((imageEstClass - labels) == 0);
acc_overall = 100*correct/length(labels);
end