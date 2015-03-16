function [data, labels] = LoadVirtualPlusRealData(param)
try
    load('./cache/data_scene.mat');
catch
    dir_virtual = param.data_dir_virtual;
    dir_real = param.data_dir_real;
    
    list_src_train = readTextFile([dir_virtual '/trainList100.txt']);
    list_tar_train = readTextFile([dir_real '/trainDAList.txt']);
    list_tar_test = readTextFile([dir_real '/evalList.txt']);
    
    fts_train_src = [];
    labels_train_src = [];
    
    for i = 1:length(list_src_train)
        % read over-segmentations (superpixels)
        segs2 = dlmread([dir_virtual '/FEATS/' list_src_train{i} '.labels.txt']);
        
        % read segmentation features
        feat2= dlmread([dir_virtual '/FEATS/' list_src_train{i} '.features.txt']);
        
        fts_train_src = [fts_train_src; feat2];
        labels_train_src = [labels_train_src; segs2];
    end
    
    inds_neg = labels_train_src == -1;
    labels_train_src(inds_neg) = [];
    fts_train_src(inds_neg, :) = [];
    
    fts_train_tar = [];
    labels_train_tar = [];
    
    for i = 1:length(list_tar_train)
        % read over-segmentations (superpixels)
        segs2 = dlmread([dir_real '/FEATS/' list_tar_train{i} '.labels.txt']);
        
        % read segmentation features
        feat2= dlmread([dir_real '/FEATS/' list_tar_train{i} '.features.txt']);
        
        fts_train_tar = [fts_train_tar; feat2];
        labels_train_tar = [labels_train_tar; segs2];
    end
    
    inds_neg = labels_train_tar == -1;
    labels_train_tar(inds_neg) = [];
    fts_train_tar(inds_neg, :) = [];
    
    fts_test_tar = [];
    labels_test_tar = [];
    
    for i = 1:length(list_tar_test)
        % read over-segmentations (superpixels)
        segs2 = dlmread([dir_real '/FEATS/' list_tar_test{i} '.labels.txt']);
        
        % read segmentation features
        feat2= dlmread([dir_real '/FEATS/' list_tar_test{i} '.features.txt']);
        
        fts_test_tar = [fts_test_tar; feat2];
        labels_test_tar = [labels_test_tar; segs2];
    end
    
    inds_neg = labels_test_tar == -1;
    labels_test_tar(inds_neg) = [];
    fts_test_tar(inds_neg, :) = [];
    
    data.train.source = fts_train_src;
    data.train.target = fts_train_tar;
    data.test.target = fts_test_tar;
    
    labels.train.source = (labels_train_src+1)';
    labels.train.target = (labels_train_tar+1)';
    labels.test.target = (labels_test_tar+1)';
    
    save('./cache/data_scene.mat', 'data', 'labels');
end
print_static_samples('Source-Train', labels.train.source);
print_static_samples('Target-Train', labels.train.target);
print_static_samples('Target-Test', labels.test.target);
end

function print_static_samples(name_dataset, labels)
num_classes = max(labels);
fprintf('\n------------------');
fprintf('\nStatistic of %s:', name_dataset);
fprintf('\nNumber of classes: %d', num_classes);
fprintf('\nTotal number of samples: %d', length(labels));
fprintf('\nNumber of samples per class:');
for i=1:num_classes
    fprintf('\nClass %d: %d', i, sum(labels==i));
end
fprintf('\n');
end