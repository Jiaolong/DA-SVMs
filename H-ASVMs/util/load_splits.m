function [train_ids, test_ids] = load_splits(src, tar, param)
result_filename = sprintf('../DataSplitsOfficeCaltech/SameCategory_%s-%s_%dRandomTrials_10Categories.mat', ...
    param.domain_names{src}, param.domain_names{tar},...
    param.num_trials);
splits    = load(result_filename);
train_ids = splits.train;
test_ids  = splits.test;
end