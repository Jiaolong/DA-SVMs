function writeTransformedSRCfeature(W, param, dir_save)


dir_train_src = [dir_save '/Train/SRC-TRANS/'];
dir_virtual = param.data_dir_virtual;

exists_or_mkdir(dir_train_src);

list_train_src = readTextFile([dir_virtual '/trainList200.txt']);

W = W(1:end-1, 1:end-1);

fprintf('\nWriting transformed features into file ...\n');

for i = 1:length(list_train_src)
    feat2 = dlmread([dir_virtual '/FEATS/' list_train_src{i} '.features.txt']);
    fprintf('SRC-TRANS: [%d/%d]\n', i, length(list_train_src));
    fname = [dir_train_src list_train_src{i} '.feat_trans.txt'];
    feat_trans = feat2*W';
    dlmwrite(fname, feat_trans, 'delimiter', ' ', 'precision', 6);
end

end