clear all;

% add dependencies
addpath('../liblinear-mmdt/matlab/');
addpath('./external/DomainTransformsECCV10/');
addpath('./SceneSegmentation/');
addpath('./MMDT/');

virtual = 1; kitti = 2; cambi = 3;
param = Config_SceneSeg(virtual, cambi);
[data, labels] = LoadVirtualPlusRealData(param);

source_domain = param.source;
target_domain = param.target;

fprintf('Source Domain - %s, Target Domain - %s\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

if param.dim < size(data.train.source, 2)
    P = princomp([data.train.source; data.train.target; data.test.target]);
    data.train.source = data.train.source * P(:, 1:param.dim);
    data.train.target = data.train.target * P(:, 1:param.dim);
    data.test.target = data.test.target * P(:, 1:param.dim);
end

% Source domain classifier
tstart = tic;
model_src = Train(labels.train, data.train, param, source_domain);
telapsed = toc(tstart);
[~, acc] = predict(labels.test.target', ...
    [sparse(data.test.target), ones(length(labels.test.target),1)], ...
    model_src);
accuracy = acc(1);
fprintf('Source domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% Target domain classifier
tstart = tic;
model_tar = Train(labels.train, data.train, param, target_domain);
telapsed = toc(tstart);
[~, acc] = predict(labels.test.target', ...
    [sparse(data.test.target), ones(length(labels.test.target),1)], ...
    model_tar);
accuracy = acc(1);
fprintf('Target domain classifier accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);

% Domain adaptation
tstart = tic;
[model_mmdt, W] = TrainMmdt(labels.train, data.train, param);
telapsed = toc(tstart);
[pl, acc] = predict(labels.test.target', ...
    [sparse(data.test.target), ones(length(labels.test.target),1)], ...
    model_mmdt);
accuracy = acc(1);
pred_labels = pl;
fprintf('After adaptation, accuracy = %6.2f (Time = %6.2f)\n', accuracy, telapsed);