function [acc_ovr, acc_avg, pl, scores, confus] = test_svm(model, labels, data, aug_data)
% Usage:
%   [acc pl] = test_svm(model, labels, data)
% Input
%   model    -trained model
%   labels   -ground truth label
%   data     -test data, num_samplesxdim
%   aug_data - if true augment data with 1
% Output:
%   acc    -accuracy
%   pl     -predicted label
%   score  - score of each sample
%   confus - confusion maxtrix
%   acc_overall - overall accuracy = total_correct/ total_sample

if nargin < 4
    aug_data = true;
end

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
if aug_data
    scores = model.w' * data + model.b' * ones(1,size(data,2));
else
    assert(size(data, 1) -1 == size(model.w, 1))
    scores = model.w' * data(1:end-1,:) + model.b'*data(end,:);
end

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
acc_avg = 100*mean(diag(confus)./numTest);
pl = imageEstClass;

correct = sum((imageEstClass - labels) == 0);
acc_ovr = 100*correct/length(labels);
end