function H = binsum(H, X, B)
% H = binsum(H,X,B) adds the elements of the array X to the
%   elements of the array H indexed by B. X and B must have the same
%   dimensions, and the elements of B must be valid indexes for the
%   array H (except for null indexes, which are silently skipped). An
%   application is the calculation of a histogram H, where B are the
%   occurences and X are the occurence weights.

sz = size(H);

H_flat = reshape(H, [], 1);
X_flat = reshape(X, [], 1);
B_flat = reshape(B, [], 1);

for i=1:length(B_flat)
    ind = B_flat(i);
    H_flat(ind) = H_flat(ind) + X_flat(i);
end

H = reshape(H_flat, sz);
end