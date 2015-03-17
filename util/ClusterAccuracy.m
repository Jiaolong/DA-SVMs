function [bestAcc bestPred]= ClusterAccuracy(pred, gt)
% Domain clustering accuracy
% Input:
%       pred  - predicted label
%       gt    - ground truth
% Output:
%       bestAcc  - best domain clustering accuracy
%       bestPred - best label assignment

classes = unique(gt);

oldTruth = gt;
for c = 1:length(classes)
   class =  classes(c);
   gt(oldTruth==class)=c;
end

n = max(pred);

possible = perms(1:n);

bestAcc = 0;
for i = 1:size(possible,1)
    tPred = pred;
    p = possible(i,:);
    for k = 1:n
        ind = find(pred==k);
        tPred(ind) = p(k)*ones(1,length(ind)); 
    end
    acc = sum(tPred == gt)/length(gt);
    if acc > bestAcc
        bestAcc = acc;
        bestPred = tPred;
    end
end

end
