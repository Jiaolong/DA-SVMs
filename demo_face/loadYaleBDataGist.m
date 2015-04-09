function [data, label] = loadYaleBDataGist(param)
% Load Yale B dataset.
% http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
%   Detailed explanation goes here

data_dir = param.data_dir;
num_tar_train = param.num_tar_train; % number of target samples per subject for DA training

data.train.source = [];
label.train.source = [];

data.train.t1 = [];
label.train.t1 = [];
data.train.t2 = [];
label.train.t2 = [];

data.test.t1 = [];
label.test.t1 = [];
data.test.t2 = [];
label.test.t2 = [];

num_folder = 39;

try
    load([param.cache_dir 'YaleB.mat']);
catch
    id_subject = 1;
    for i = 1 : num_folder
        folder = sprintf('yaleB%02d', i);
        
        if ~exist([data_dir folder], 'dir')
            continue;
        end
        
        fs_dark = [];
        n_dark = 1;
        fs_light = [];
        n_light = 1;
        fs_normal = [];
        n_normal = 1;
        
        fts_dark = [];
        lbs_dark = [];
        fts_light = [];
        lbs_light = [];
        fts_normal = [];
        lbs_normal = [];

        fs = dir([data_dir folder '/*.pgm']);
        for j = 1 : length(fs)
            fname = fs(j).name;
            
            % yaleB01_P00_Ambient.pgm
            if ismember('Ambient', fname)
                continue;
            end
            
            % yaleB01_P00A-005E-10.pgm
            a = str2num(fname(13:16));
            e = str2num(fname(18:20));
            
            lb = id_subject;
            im = imread([data_dir folder '/' fname]);
            gs = double(computeGist(im));
            
            if abs(a) > 85 || abs(e) > 40
                fts_dark = [fts_dark; gs];
                lbs_dark = [lbs_dark; lb];
                fs_dark{n_dark} = fname;
                n_dark = n_dark + 1;
            elseif abs(a) < 25 && abs(e) < 25
                fts_light = [fts_light; gs];
                lbs_light = [lbs_light; lb];
                fs_light{n_light} = fname;
                n_light = n_light + 1;
            else
                fts_normal = [fts_normal; gs];
                lbs_normal = [lbs_normal; lb];
                fs_normal{n_normal} = fname;
                n_normal = n_normal + 1;
            end
        end
        
        data.train.source  = [data.train.source; fts_normal];
        label.train.source = [label.train.source; lbs_normal];

        data.train.t1 = [data.train.t1; fts_dark(1:num_tar_train,:)];
        label.train.t1 = [label.train.t1; lbs_dark(1:num_tar_train)];
        data.test.t1 = [data.test.t1; fts_dark(num_tar_train+1:end,:)];
        label.test.t1 = [label.test.t1; lbs_dark(num_tar_train+1:end)];
        
        data.train.t2 = [data.train.t2; fts_light(1:num_tar_train,:)];
        label.train.t2 = [label.train.t2; lbs_light(1:num_tar_train)];
        data.test.t2 = [data.test.t2; fts_light(num_tar_train+1:end,:)];
        label.test.t2 = [label.test.t2; lbs_light(num_tar_train+1:end)];
        
        id_subject = id_subject + 1;      
    end
    
    save([param.cache_dir 'YaleB.mat'], 'data', 'label');
end
end

