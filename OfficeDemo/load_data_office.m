function [Data, Labels, Images] = load_data_office(foldername, norm_type)
% LoadOfficePlusCaltechData
fname = '%s_SURF_L10.mat';
fname_image = '%s_SURF_L10_imgs.mat';
domain_names = {'amazon', 'webcam', 'dslr', 'Caltech10'};

Data = cell(numel(domain_names));
Labels = cell(numel(domain_names));
Images = cell(numel(domain_names));
for d = 1:numel(domain_names)
   fullfilename = fullfile(foldername, sprintf(fname, domain_names{d}));
   fullfilename_imgs = fullfile(foldername, sprintf(fname_image, domain_names{d}));
   load(fullfilename);
   load(fullfilename_imgs);
   fts = NormData(fts, norm_type);
   imgNames  = Parse_names(imgNames);
   Data{d}   = fts;
   Labels{d} = labels';
   Images{d} = imgNames;
end
end

function imgNames  = Parse_names(imgNames)
for i=1:length(imgNames)
    img_name = imgNames{i};
    index = find(img_name=='_');
    img_name(index(1))='/';
    img_name(index(end-1))='/';
    imgNames{i} = [img_name '.jpg'];
end
end
