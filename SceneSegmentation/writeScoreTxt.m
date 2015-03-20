function writeScoreTxt(scores, param, dir_save)

exists_or_mkdir(dir_save);
dir_train_src = [dir_save '/Train/SRC/'];
dir_train_tar = [dir_save '/Train/TAR/'];
dir_test_tar = [dir_save '/Test/TAR/'];

exists_or_mkdir(dir_train_src);
exists_or_mkdir(dir_train_tar);
exists_or_mkdir(dir_test_tar);

dir_virtual = param.data_dir_virtual;
dir_real = param.data_dir_real;

list_train_src = readTextFile([dir_virtual '/trainList200.txt']);
list_train_tar = readTextFile([dir_real '/trainDAList.txt']);
list_test_tar = readTextFile([dir_real '/evalList.txt']);

num_class = param.num_class;
fprintf('\nWriting scores into file ...\n');
% TRAIN-SRC
if isfield(scores.train, 'source')
    n = 1;
    for i = 1:length(list_train_src)
        feat2 = dlmread([dir_virtual '/FEATS/' list_train_src{i} '.features.txt']);
        segs2 = dlmread([dir_virtual '/FEATS/' list_train_src{i} '.labels.txt']);
        fprintf('TRAIN-SRC: [%d/%d]\n', i, length(list_train_src));
        fname = [dir_train_src list_train_src{i} '.scores.txt'];
        n = writeOneFile(fname, scores.train.source, n, feat2, segs2, num_class);
    end
end

% TRAIN-TAR
if isfield(scores.train, 'target')
    n = 1;
    for i = 1:length(list_train_tar)
        feat2 = dlmread([dir_real '/FEATS/' list_train_tar{i} '.features.txt']);
        segs2 = dlmread([dir_real '/FEATS/' list_train_tar{i} '.labels.txt']);
        fprintf('TRAIN-TAR: [%d/%d]\n', i, length(list_train_tar));
        fname = [dir_train_tar list_train_tar{i} '.scores.txt'];
        n = writeOneFile(fname, scores.train.target, n, feat2, segs2, num_class);
    end
end

% TEST-TAR
if isfield(scores.test, 'target')
    n = 1;
    for i = 1:length(list_test_tar)
        feat2 = dlmread([dir_real '/FEATS/' list_test_tar{i} '.features.txt']);
        segs2 = dlmread([dir_real '/FEATS/' list_test_tar{i} '.labels.txt']);
        fprintf('TEST-TAR: [%d/%d]\n', i, length(list_test_tar));
        fname = [dir_test_tar list_test_tar{i} '.scores.txt'];
        n = writeOneFile(fname, scores.test.target, n, feat2, segs2, num_class);
    end
end
end

function n = writeOneFile(fname, s, n, feat2, segs2, num_class)
fid = fopen(fname, 'w');
for j = 1 : size(feat2, 1)
    if segs2(j) == -1
        s_ij = zeros(num_class, 1);
    else
        s_ij = s(:, n);
        n = n + 1;
    end
    fprintf(fid, '%6.2f ', s_ij');
    fprintf(fid, '\n');
end
fclose(fid);
end