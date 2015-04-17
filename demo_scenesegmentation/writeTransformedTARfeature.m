function writeTransformedTARfeature(W, param, dir_save)
% Write transformed target domain features into files

dir_FEAT_TRANS = [dir_save '/FEATS-TRANS/'];
dir_target = param.data_dir_real;

exists_or_mkdir(dir_FEAT_TRANS);

list_train_tar = readTextFile(param.train_tar_list);

fprintf('\nWriting transformed features into file ...\n');

for i = 1:length(list_train_tar)
    feat = dlmread([dir_target '/FEATS/' list_train_tar{i} '.features.txt']);
    fprintf('TRAN-TAR-TRAIN: [%d/%d]\n', i, length(list_train_tar));
    fname = [dir_FEAT_TRANS list_train_tar{i} '.features.txt'];
    feat2 = [feat, ones(size(feat,1), 1)];
    feat_trans = feat2*W;
    feat_trans(:,end) = [];
    dlmwrite(fname, feat_trans, 'delimiter', ' ', 'precision', 6);
end

list_test_tar = readTextFile(param.test_tar_list);

for i = 1:length(list_test_tar)
    feat = dlmread([dir_target '/FEATS/' list_test_tar{i} '.features.txt']);
    fprintf('TRAN-TAR-TEST: [%d/%d]\n', i, length(list_test_tar));
    fname = [dir_FEAT_TRANS list_test_tar{i} '.features.txt'];
    feat2 = [feat, ones(size(feat,1), 1)];
    feat_trans = feat2*W;
    feat_trans(:,end) = [];
    dlmwrite(fname, feat_trans, 'delimiter', ' ', 'precision', 6);
end
end