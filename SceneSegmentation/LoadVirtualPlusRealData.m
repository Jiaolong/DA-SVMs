function [data, labels] = LoadVirtualPlusRealData(param)

dir_virtual = param.data_dir_virtual;
dir_real = param.data_dir_real;

list_src_train = readTextFile([dir_virtual '/trainList200.txt']);
list_tar_train = readTextFile([dir_real '/trainDAList.txt']);
list_tar_test = readTextFile([dir_real '/evalList.txt']);

fts_train_src = [];
labels_train_src = [];

for i = 1:length(list_src_train)
    % read over-segmentations (superpixels)
    segs2 = dlmread([dir_virtual '/FEATS/' list_src_train{i} '.labels.txt']);
    % so that they start with index 1:
    segs2=segs2+1;
    assert(min(segs2(:))>=0);
    % read segmentation features
    feat2= dlmread([dir_virtual '/FEATS/' list_src_train{i} '.features.txt']);
    
    fts_train_src = [fts_train_src; feat2];
    labels_train_src = [labels_train_src; segs2];
end

fts_train_tar = [];
labels_train_tar = [];

for i = 1:length(list_tar_train)
    % read over-segmentations (superpixels)
    segs2 = dlmread([dir_real '/FEATS/' list_tar_train{i} '.labels.txt']);
    % so that they start with index 1:
    segs2=segs2+1;
    assert(min(segs2(:))>=0)
    % read segmentation features
    feat2= dlmread([dir_real '/FEATS/' list_tar_train{i} '.features.txt']);
    
    fts_train_tar = [fts_train_tar; feat2];
    labels_train_tar = [labels_train_tar; segs2];
end

fts_test_tar = [];
labels_test_tar = [];

for i = 1:length(list_tar_test)
    % read over-segmentations (superpixels)
    segs2 = dlmread([dir_real '/FEATS/' list_tar_test{i} '.labels.txt']);
    % so that they start with index 1:
    segs2=segs2+1;
    assert(min(segs2(:))>=0)
    % read segmentation features
    feat2= dlmread([dir_real '/FEATS/' list_tar_test{i} '.features.txt']);
    
    fts_test_tar = [fts_test_tar; feat2];
    labels_test_tar = [labels_test_tar; segs2];
end

data.train.source = fts_train_src;
data.train.target = fts_train_tar;
data.test.target = fts_test_tar;

labels.train.source = (labels_train_src+1)';
labels.train.target = (labels_train_tar+1)';
labels.test.target = (labels_test_tar+1)';
end