function [acc, pred] = KernelKnnClassify(Ytrain, Xtrain, Ytest, Xtest)
K = Xtrain * Xtest';
nKtrain = diag(Xtrain*Xtrain');
num_te = numel(Ytest);
nKtest = diag(Xtest * Xtest');
pred = kernelKNN(Ytrain, K', nKtrain, nKtest, 1)';
acc = 100 * sum(pred == Ytest) / num_te;
end