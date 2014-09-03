function means = KmeansPlusPlus(X, k, start)
% X - data points [n x d] (n points of d dimension)
% k - number of clusters to create
% start = 0 --> random initialization
% start ~= 0 --> kmeans++ intialization
if start == 0
    means = rand(k, size(X, 2));
else
    num_pts = size(X, 1);
    means = zeros(k, size(X, 2));
    rind = randi(num_pts);
    % randomly assign the first mean to a data-point
    means(1,:) = X(rind, :); %X(cType, :,class);
    
    for j = 2:k
        D = zeros(num_pts, 1); %distances
        for i = 1:length(D)
           % compute D(i)=distance b/t pt i and nearest cluster already
           % chosen
           d = zeros(k,1);
           for source = 1:k
                d(source) = norm(means(source,:) - X(i, :));
           end
           D(i) = min(d);
        end
        %choose new pt as random mean according to D(x)^2
        D2 = D.^2;
        val = rand()*sum(D2);
        s= 0;
        i = 1;
        while (s < val) && (i < length(D))
            s = s + D2(i);
            i = i +1;
        end
        means(j,:) = X(i, :);
    end  
end
end
