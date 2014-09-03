function view_domains
% view_domains
%   visulize the discovered domains

plot_hist = true;
a = 1; w = 2; d = 3; c = 4;
target_domains = [w d c];
param = config();
[~, Labels, Images] = load_data(param.DATA_DIR, param.norm_type);
Y = cat(2, Labels{target_domains});
I = cat(1, Images{target_domains});
% load discovered domains
str_domains = cat(2, param.domain_abrv{target_domains});
fname = ['./data_save/latent_domains_nips/domain_index/'...
    'latent_domains_' str_domains '_pr.mat'];
load(fname);
z1 = z;
acc1 = acc;
Labels_1 = cell(3,1);
Images_1 = cell(3,1);
for i=1:3
    Labels_1{i} = Y(z1 == i);
    Images_1{i} = I(z1 == i);
end
% path_save = ['./cache/img_latent_doamin_nips_pr_' str_domains];
% save_images(param, Images_1, Labels_1, path_save);

fname = ['./data_save/latent_domains_nips/domain_index/'...
    'latent_domains_' str_domains '_.mat'];
load(fname);
z2 = z;
acc2 = acc;
Labels_2 = cell(3,1);
for i=1:3
    Labels_2{i} = Y(z2 == i);
end

fname = ['./data_save/latent_domains_eccv/domain_index/'...
    'latent_domains_' str_domains '_pr.mat'];
load(fname);
z3 = z;
acc3 = acc;
Labels_3 = cell(3,1);
for i=1:3
    Labels_3{i} = Y(z3 == i);
end

fname = ['./data_save/latent_domains_eccv/domain_index/'...
    'latent_domains_' str_domains '_.mat'];
load(fname);
z4 = z;
acc4 = acc;
Labels_4 = cell(3,1);
for i=1:3
    Labels_4{i} = Y(z4 == i);
end

% ground truth domain labels
D = [];
for i=1:length(target_domains)
    d = target_domains(i);
    D = [D; i*ones(length(Labels{d}),1)];
end

% visualization
width = 15;
im_gt = repmat(D, 1, width);
im_z1 = repmat(z1, 1, width);
im_z2 = repmat(z2, 1, width);
im_z3 = repmat(z3, 1, width);
im_z4 = repmat(z4, 1, width);
% colormap('Lines');
n_rows   = 1;
n_clomns = 5;
if plot_hist
    n_rows   = 4;
    n_clomns = 5;
end

subplot(n_rows,n_clomns,1);
imagesc(im_gt);
set(gca,'XTick',[], 'XColor',[0 0 0]);
title('Ground truth','FontSize',20);

subplot(n_rows,n_clomns,2);
imagesc(im_z1);
set(gca,'XTick',[], 'XColor',[0 0 0]);
set(gca,'YTick',[], 'YColor',[0 0 0]);
title(sprintf('Reshape(Pr) [%0.2f] ', acc1),'FontSize',20);

subplot(n_rows,n_clomns,3);
imagesc(im_z2);
set(gca,'XTick',[], 'XColor',[0 0 0]);
set(gca,'YTick',[], 'YColor',[0 0 0]);
title(sprintf('Reshape [%0.2f] ', acc2),'FontSize',20);

subplot(n_rows,n_clomns,4);
imagesc(im_z3);
set(gca,'XTick',[], 'XColor',[0 0 0]);
set(gca,'YTick',[], 'YColor',[0 0 0]);
title(sprintf('LatDD(Pr) [%0.2f] ', acc3),'FontSize',20);

subplot(n_rows,n_clomns,5);
imagesc(im_z4);
set(gca,'XTick',[], 'XColor',[0 0 0]);
set(gca,'YTick',[], 'YColor',[0 0 0]);
title(sprintf('LatDD [%0.2f] ', acc4),'FontSize',20);

if plot_hist
    for i=1:length(target_domains)
    d = target_domains(i);
    subplot(n_rows,n_clomns,5+(i-1)*5 +1);
    % compute class distribution
    hist_class(Labels{d});
    end

    for i=1:length(target_domains)
        subplot(n_rows,n_clomns,6+(i-1)*5 +1);
        % compute class distribution
        hist_class(Labels_1{i});
    end
    
    for i=1:length(target_domains)
        subplot(n_rows,n_clomns,7+(i-1)*5 +1);
        % compute class distribution
        hist_class(Labels_2{i});
    end
    
    for i=1:length(target_domains)
        subplot(n_rows,n_clomns,8+(i-1)*5 +1);
        % compute class distribution
        hist_class(Labels_3{i});
    end
    
    for i=1:length(target_domains)
        subplot(n_rows,n_clomns,9+(i-1)*5 +1);
        % compute class distribution
        hist_class(Labels_4{i});
    end
end
end

function save_images(param, Images, Labels, path_save)
dir_images = param.IMAGE_DIR;
for i=1:size(Labels,1)
    % creat domain folder
    path_d = [path_save '/domain_' num2str(i)];
    labels_d = Labels{i};
    images_d = Images{i};
    C = unique(labels_d);
    for j=1:length(C)
        c_name = param.categories{C(j)};
        path_d_c = [path_d '/' c_name '/'];
        mkdir(path_d_c);
        images_d_c = images_d(labels_d==C(j));
        for k=1:size(images_d_c,1)
            copyfile([dir_images images_d_c{k}], path_d_c);
        end
    end
end
end

function hist_class(Labels_d)
C = unique(Labels_d);
[f,x] = hist(Labels_d, C);
bar(x,f/sum(f));
end