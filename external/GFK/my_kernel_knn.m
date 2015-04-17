function [prediction accuracy] = my_kernel_knn(M, Xr, Yr, Xt, Yt)
dist = repmat(diag(Xr*M*Xr'),1,length(Yt)) ...
    + repmat(diag(Xt*M*Xt')',length(Yr),1)...
    - 2*Xr*M*Xt';
[~, minIDX] = min(dist);
prediction = Yr(minIDX);
accuracy = sum( prediction==Yt ) / length(Yt); 