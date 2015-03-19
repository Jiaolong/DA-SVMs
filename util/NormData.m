function fts = NormData(fts, norm_type)
    switch norm_type
        case 'l1_zscore'
            fts = fts ./ repmat(sum(abs(fts),2),1,size(fts,2));
            fts = zscore(fts,1);
        case 'l2_zscore'
            fts = fts ./ repmat(sqrt(sum(fts.^2,2)),1,size(fts,2));
            fts = zscore(fts,1);
        case 'l1'
            fts = fts ./ repmat(sum(abs(fts),2),1,size(fts,2));
        case 'l2'
            fts = fts ./ repmat(sqrt(sum(fts.^2,2)),1,size(fts,2));
        case 'none'
            return;
        otherwise
            error('norm');
    end
end