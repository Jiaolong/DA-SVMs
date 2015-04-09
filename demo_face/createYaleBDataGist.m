function [Gists] = createYaleBDataGist(Path1)
    Path1e = sprintf('%s/*.jpg', Path1);
    
    files = dir(Path1e);
    
    Data = cell(length(files), 1);
    for i=1:length(files)
       I = imread([Path1 '/' files(i).name]); 
       
       gs = computeGist(I);
       Data{i} = double(gs);
       
       str0 = sprintf('Read image %d / %d', i, length(files));
       disp(str0);
    end
    
    Gists = cell2mat(Data);
end