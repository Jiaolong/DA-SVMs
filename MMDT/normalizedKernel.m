function K = normalizedKernel(K)
if size(K,1) == size(K,2) %square
        DD = abs(diag(K));
        sq_DD = sqrt(DD);
        sq_DD2 = sq_DD * sq_DD';
        K = K ./ sq_DD2;
    end
end
