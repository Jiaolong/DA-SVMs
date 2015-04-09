function gs = computeGist(Im)
    % Parameters:
    clear param
    param.imageSize = size(Im); % it works also with non-square images
    param.orientationsPerScale = [8 8 8 8 8];
    param.numberBlocks = 12;
    param.fc_prefilt = 4;

    % Computing gist requires 1) prefilter image, 2) filter image and collect
    % output energies
    [gs, param] = LMgist(Im, '', param);
end
