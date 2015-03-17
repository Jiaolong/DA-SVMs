function [acc pl scores confus, acc_overall] = test_svm(model, labels, data, param)
% Usage:
%   [acc pl] = test_svm(model, labels, data, param)
% Input
%   model  -trained model
%   labels -ground truth label
%   data   -test data, num_samplesxdim
%   param  -the parameters
% Output:
%   acc    -accuracy
%   pl     -predicted label

classes = param.categories;
data = data';
numTest = zeros(length(classes), 1);
numTest = binsum(numTest, ones(length(labels),1), labels);
numTest(numTest == 0) = Inf;
% Estimate the class of the test images
scores = model.w' * data + model.b' * ones(1,size(data,2));

[~, imageEstClass] = max(scores, [], 1);

% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
    labels, imageEstClass);
confus = zeros(length(classes));
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