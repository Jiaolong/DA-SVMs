function data = NormalizeData(data, use_norm)
if use_norm
    for i = 1:size(data,1)
        data(i,:) = data(i,:) / norm(data(i,:));
    end
end
end