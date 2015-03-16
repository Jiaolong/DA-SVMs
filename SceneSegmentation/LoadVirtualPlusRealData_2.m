function [data, labels] = LoadVirtualPlusRealData_2(foldername)

num_tar_train = 5;

data_train = load(fullfile(foldername, 'kitti-allNeighborPairs_train'));
data_test = load(fullfile(foldername, 'kitti-allNeighborPairs_eval'));
all_data_train = data_train.allData;
all_data_test = data_test.allData;

fts_train_src = [];
labels_train_src = [];
fts_train_tar = [];
labels_train_tar = [];

for i = 1:length(all_data_train)
    fts_i = all_data_train{i}.feat2;
    labels_i = all_data_train{i}.segLabels;
    
    
    if i > num_tar_train
        fts_train_src = [fts_train_src;fts_i];
        labels_train_src = [labels_train_src;labels_i];
    else
        fts_train_tar = [fts_train_tar;fts_i];
        labels_train_tar = [labels_train_tar;labels_i];
    end
end

fts_test_tar = [];
labels_test_tar = [];

for i = 1:length(all_data_test)
    fts_i = all_data_test{i}.feat2;
    labels_i = all_data_test{i}.segLabels;
    fts_test_tar = [fts_test_tar;fts_i];
    labels_test_tar = [labels_test_tar;labels_i];
end

data.train.source = fts_train_src;
data.train.target = fts_train_tar;
data.test.target = fts_test_tar;

labels.train.source = (labels_train_src+1)';
labels.train.target = (labels_train_tar+1)';
labels.test.target = (labels_test_tar+1)';
end