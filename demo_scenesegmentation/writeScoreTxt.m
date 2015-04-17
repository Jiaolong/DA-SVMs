function writeScoreTxt(scores, param, dir_save, model, feat_aug)
% write the scores into txt files
% feat_aug - feature augmentation

if nargin < 5
    feat_aug = false;
else
    feat_aug = true;
end

dir_scores = [dir_save '/SCORES/'];
exists_or_mkdir(dir_scores);

dir_virtual = param.data_dir_virtual;
dir_real = param.data_dir_real;

list_train_src = readTextFile(param.train_src_list);
list_train_tar = readTextFile(param.train_tar_list);
list_test_tar = readTextFile(param.test_tar_list);

ext = '.boosted.txt';
model = order_model(model);

fprintf('\nWriting scores into file ...\n');
% TRAIN-SRC
if isfield(scores.train, 'source')
    n = 1;
    for i = 1:length(list_train_src)
        feat2 = dlmread([dir_virtual '/FEATS/' list_train_src{i} '.features.txt']);
        segs2 = dlmread([dir_virtual '/SuperPixelsLabels/' list_train_src{i} '.labels.txt']);
        fprintf('TRAIN-SRC: [%d/%d]\n', i, length(list_train_src));
        fname = [dir_scores list_train_src{i} ext];
        n = writeOneFile(fname, scores.train.source, n, feat2, segs2, model, feat_aug);
    end
end

% TRAIN-TAR
if isfield(scores.train, 'target')
    n = 1;
    for i = 1:length(list_train_tar)
        feat2 = dlmread([dir_real '/FEATS/' list_train_tar{i} '.features.txt']);
        segs2 = dlmread([dir_real '/SuperPixelsLabels/' list_train_tar{i} '.labels.txt']);
        fprintf('TRAIN-TAR: [%d/%d]\n', i, length(list_train_tar));
        fname = [dir_scores list_train_tar{i} ext];
        n = writeOneFile(fname, scores.train.target, n, feat2, segs2, model, feat_aug);
    end
end

% TEST-TAR
if isfield(scores.test, 'target')
    n = 1;
    for i = 1:length(list_test_tar)
        feat2 = dlmread([dir_real '/FEATS/' list_test_tar{i} '.features.txt']);
        segs2 = dlmread([dir_real '/SuperPixelsLabels/' list_test_tar{i} '.labels.txt']);
        fprintf('TEST-TAR: [%d/%d]\n', i, length(list_test_tar));
        fname = [dir_scores list_test_tar{i} ext];
        n = writeOneFile(fname, scores.test.target, n, feat2, segs2, model, feat_aug);
    end
end
end

function n = writeOneFile(fname, s, n, feat2, segs2, model, feat_aug)
s_mat = [];
for j = 1 : size(feat2, 1)
    if segs2(j) == -1
        if feat_aug
            data = [feat2(j,:), zeros(1, size(feat2, 2)*2)]';
        else
            data = feat2(j, :)';
        end
        s_ij = model.w' * data + model.b' * ones(1,size(data,2));
    else
        s_ij = s(:, n);
        n = n + 1;
    end
    s_mat = [s_mat; s_ij'];
end
dlmwrite(fname, s_mat, 'delimiter', ' ', 'precision', 6);
end