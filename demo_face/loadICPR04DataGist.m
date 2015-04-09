function [data, label] = loadICPR04DataGist(param)
% ICPR'04 benchmark dataset
%   http://www-prima.inrialpes.fr/Pointing04/data-face.html

data_dir = param.data_dir;
num_subjects = param.num_class;
num_tar_train = param.num_tar_train; % number of target samples per subject for DA training

VerticalAngle = {'-90', '-60', '-30', '-15', '+0', '+15', '+30', '+60', '+90'};
HorizontalAngle = {'-90', '-75', '-60', '-45', '-30', '-15', '+0',...
    '+15', '+30', '+45', '+60', '+75', '+90'};

tar1_h_angles = HorizontalAngle(1: 2); % left -90, -75
tar2_h_angles = HorizontalAngle(3: 4); % left -60, -45
src_h_angles = HorizontalAngle(5: 9); % frontal
tar3_h_angles = HorizontalAngle(10: 11); % left -90, -75
tar4_h_angles = HorizontalAngle(12: 13); % left -60, -45

ft_tar1 = [];
label_tar1 = [];
ft_tar2 = [];
label_tar2 = [];
ft_tar3 = [];
label_tar3 = [];
ft_tar4 = [];
label_tar4 = [];

try
    load([param.cache_dir '/icpr04_gist_all.mat']);
catch
    
    % frontal
    n = 1;
    folder = 'deFace';
    for i = 1 : num_subjects
        % sequence 1
        fname = sprintf('personne%02d146+0+0.jpg', i);
        full_name = [data_dir '/' folder '/' fname];
        im = imread(full_name);
        
        gs = computeGist(im);
        Data{n} = double(gs);
        Ls(n) = i;
        n = n + 1;
        
        % sequence 2
        fname = sprintf('personne%02d246+0+0.jpg', i);
        full_name = [data_dir '/' folder '/' fname];
        im = imread(full_name);
        
        gs = computeGist(im);
        Data{n} = double(gs);
        Ls(n) = i;
        n = n + 1;
    end
    
    ft_src = reshape(cell2mat(Data), length(Data), []);
    label_src = Ls';
    
    
    for i = 1 : num_subjects
        folder = sprintf('Personne%02d', i);
        fs = dir([data_dir '/' folder '/personne*.jpg']);
        for j = 1 : length(fs)
            
            fname = fs(j).name;
            h_angles = fname(14:end-4);
            
            im = imread([data_dir '/' folder '/' fname]);
            gs = double(computeGist(im));
            
            if ismember(h_angles(end-1:end), src_h_angles) ||...
                    ismember(h_angles(end-2:end), src_h_angles)
                ft_src = [ft_src; gs];
                label_src = [label_src; i];
            elseif ismember(h_angles(end-1:end), tar1_h_angles) ||...
                    ismember(h_angles(end-2:end), tar1_h_angles)
                ft_tar1 = [ft_tar1; gs];
                label_tar1 = [label_tar1; i];
            elseif ismember(h_angles(end-1:end), tar2_h_angles) ||...
                    ismember(h_angles(end-2:end), tar2_h_angles)
                ft_tar2 = [ft_tar2; gs];
                label_tar2 = [label_tar2; i];
            elseif ismember(h_angles(end-1:end), tar3_h_angles) ||...
                    ismember(h_angles(end-2:end), tar3_h_angles)
                ft_tar3 = [ft_tar3; gs];
                label_tar3 = [label_tar3; i];
            elseif ismember(h_angles(end-1:end), tar4_h_angles) ||...
                    ismember(h_angles(end-2:end), tar4_h_angles)
                ft_tar4 = [ft_tar4; gs];
                label_tar4 = [label_tar4; i];
            end
        end
    end
    
    save([param.cache_dir '/icpr04_gist_all.mat'], 'ft_src', 'label_src',...
        'ft_tar1', 'label_tar1', 'ft_tar2', 'label_tar2', 'ft_tar3',...
        'label_tar3', 'ft_tar4', 'label_tar4');
end

fs_src = ...
    get_filename_list(data_dir, num_subjects, src_h_angles);

assert(length(fs_src) == size(ft_src,1));

inds_1 = inds_horizontal_angle(fs_src, '+30');
inds_2 = inds_horizontal_angle(fs_src, '-30');
inds_3 = inds_horizontal_angle(fs_src, '+15');
inds_4 = inds_horizontal_angle(fs_src, '-15');
inds = [inds_1; inds_2; inds_3; inds_4];
ft_src(inds,:) = [];
label_src(inds) = [];

data.train.source = ft_src;
label.train.source = label_src;

% split train/test in target domain
ft_t1_train = [];
ft_t1_test = [];
lbs_t1_train = [];
lbs_t1_test = [];

ft_t2_train = [];
ft_t2_test = [];
lbs_t2_train = [];
lbs_t2_test = [];

ft_t3_train = [];
ft_t3_test = [];
lbs_t3_train = [];
lbs_t3_test = [];

ft_t4_train = [];
ft_t4_test = [];
lbs_t4_train = [];
lbs_t4_test = [];

for i = 1 : num_subjects
    % T1
    inds_subject = find(label_tar1 == i);
    inds_train = inds_subject(1:num_tar_train);
    inds_test = inds_subject(num_tar_train+1:end);
    ft_t1_train = [ft_t1_train; ft_tar1(inds_train,:)];
    ft_t1_test = [ft_t1_test; ft_tar1(inds_test,:)];
    lbs_t1_train = [lbs_t1_train; label_tar1(inds_train)];
    lbs_t1_test = [lbs_t1_test; label_tar1(inds_test)];
    
    % T2
    inds_subject = find(label_tar2 == i);
    inds_train = inds_subject(1:num_tar_train);
    inds_test = inds_subject(num_tar_train+1:end);
    ft_t2_train = [ft_t2_train; ft_tar2(inds_train,:)];
    ft_t2_test = [ft_t2_test; ft_tar2(inds_test,:)];
    lbs_t2_train = [lbs_t2_train; label_tar2(inds_train)];
    lbs_t2_test = [lbs_t2_test; label_tar2(inds_test)];
    
    % T3
    inds_subject = find(label_tar3 == i);
    inds_train = inds_subject(1:num_tar_train);
    inds_test = inds_subject(num_tar_train+1:end);
    ft_t3_train = [ft_t3_train; ft_tar3(inds_train,:)];
    ft_t3_test = [ft_t3_test; ft_tar3(inds_test,:)];
    lbs_t3_train = [lbs_t3_train; label_tar3(inds_train)];
    lbs_t3_test = [lbs_t3_test; label_tar3(inds_test)];
    
    % T4
    inds_subject = find(label_tar4 == i);
    inds_train = inds_subject(1:num_tar_train);
    inds_test = inds_subject(num_tar_train+1:end);
    ft_t4_train = [ft_t4_train; ft_tar4(inds_train,:)];
    ft_t4_test = [ft_t4_test; ft_tar4(inds_test,:)];
    lbs_t4_train = [lbs_t4_train; label_tar4(inds_train)];
    lbs_t4_test = [lbs_t4_test; label_tar4(inds_test)];
end

data.train.t1 = ft_t1_train;
data.train.t2 = ft_t2_train;
data.train.t3 = ft_t3_train;
data.train.t4 = ft_t4_train;
label.train.t1 = lbs_t1_train;
label.train.t2 = lbs_t2_train;
label.train.t3 = lbs_t3_train;
label.train.t4 = lbs_t4_train;

data.test.t1 = ft_t1_test;
data.test.t2 = ft_t2_test;
data.test.t3 = ft_t3_test;
data.test.t4 = ft_t4_test;
label.test.t1 = lbs_t1_test;
label.test.t2 = lbs_t2_test;
label.test.t3 = lbs_t3_test;
label.test.t4 = lbs_t4_test;
end

function [fs_src, fs_t1, fs_t2, fs_t3, fs_t4] = ...
    get_filename_list(data_dir, num_subjects, src_h_angles, ...
    tar1_h_angles, tar2_h_angles, tar3_h_angles, tar4_h_angles)

fs_src = [];
fs_t1 = [];
fs_t2 = [];
fs_t3 = [];
fs_t4 = [];

% frontal
n_src = 1;
for i = 1 : num_subjects
    % sequence 1
    fname = sprintf('personne%02d146+0+0.jpg', i);
    fs_src{n_src} = fname;
    n_src = n_src + 1;
    
    % sequence 2
    fname = sprintf('personne%02d246+0+0.jpg', i);
    fs_src{n_src} = fname;
    n_src = n_src + 1;
end

n_t1 = 1;
for i = 1 : num_subjects
    folder = sprintf('Personne%02d', i);
    fs = dir([data_dir '/' folder '/personne*.jpg']);
    for j = 1 : length(fs)
        
        fname = fs(j).name;
        h_angles = fname(14:end-4);
        
        if ismember(h_angles(end-1:end), src_h_angles) ||...
                ismember(h_angles(end-2:end), src_h_angles)
            fs_src{n_src} = fname;
            n_src = n_src + 1;
        elseif nargin > 4 && (ismember(h_angles(end-1:end), tar1_h_angles) ||...
                ismember(h_angles(end-2:end), tar1_h_angles))
            fs_t1{n_t1} = fname;
            n_t1 = n_t1 + 1;
        elseif nargin > 5 && (ismember(h_angles(end-1:end), tar2_h_angles) ||...
                ismember(h_angles(end-2:end), tar2_h_angles))
            fs_t2{n_t2} = fname;
            n_t2 = n_t2 + 1;
        elseif nargin > 6 && (ismember(h_angles(end-1:end), tar3_h_angles) ||...
                ismember(h_angles(end-2:end), tar3_h_angles))
            fs_t3{n_t3} = fname;
            n_t3 = n_t3 + 1;
        elseif nargin > 7 && (ismember(h_angles(end-1:end), tar4_h_angles) ||...
                ismember(h_angles(end-2:end), tar4_h_angles))
            fs_t4{n_t4} = fname;
            n_t4 = n_t4 + 1;
        end
    end
end
end

function inds = inds_horizontal_angle(fs, str_angle)
% find the index of the files with specific angle
inds = [];
for i = 1 : length(fs)
    fname = fs{i};
    h_angles = fname(14:end-4);
    
    if strcmp(h_angles(end-1:end), str_angle) ||...
                strcmp(h_angles(end-2:end), str_angle)
            inds = [inds; i];
    end
end
end
