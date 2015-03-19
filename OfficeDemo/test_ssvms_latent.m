% This file is deprecated

% clear all;
% startup;
% 
% addpath(genpath('domain_cluster_tree'));
% addpath('util');
% 
% param = config();
% [Data, Labels] = load_data(param.DATA_DIR, param.norm_type);
% 
% amazon = 1; webcam = 2; dslr = 3; caltech = 4;
% % source_domain  = param.source; 
% % target_domain  = param.target; 
% source_domain  = 2; 
% target_domain  = 1;
% param = config(source_domain, target_domain);
% target_domains = param.domains(param.domains ~= source_domain); 
% 
% % Store results:
% n = param.num_trials;
% telapsed_ssvm  = zeros(n,1);
% accuracy_ssvm  = zeros(n,1);
% telapsed_assvm = zeros(n,1);
% accuracy_assvm = zeros(n,1);
% telapsed_hasvm = zeros(n,1);
% accuracy_hasvm = zeros(n,1);
% 
% fprintf('       Iteration: %d', n);
% for i = 1:n
%     fprintf('...%d', n-i);
%     % Load data splits
%     [train_ids, test_ids] = load_splits(source_domain, target_domain, param);
%     data.train.source = Data{source_domain}(train_ids.source{i}, :);
%     data.train.target = Data{target_domain}(train_ids.target{i}, :);
%     data.test.target  = Data{target_domain}(test_ids.target{i}, :);
%     
%     labels.train.source = Labels{source_domain}(train_ids.source{i});
%     labels.train.target = Labels{target_domain}(train_ids.target{i});
%     labels.test.target  = Labels{target_domain}(test_ids.target{i});
%     labels = update_labels(labels, param);
%     
%     % T1
%     data.train.targets = data.train.target;
%     data.train.target_model_ids = DEF_MODEL_IDS.M_T1*ones(length(labels.train.target),1);
%     labels.train.targets = labels.train.target;
%     model_test_id = DEF_MODEL_IDS.M_T1;
%     
%     data.test.targets = data.test.target;
%     data.test.target_model_ids = DEF_MODEL_IDS.M_T1*ones(length(labels.test.target),1);
%     labels.test.targets = labels.test.target;
%     % T2, T3
%     count = 0;
%     for j=1:length(target_domains)
%         t_id = target_domains(j);
%         if t_id == target_domain
%             continue;
%         end
%         count = count + 1;
%         m_id =  DEF_MODEL_IDS.M_T1 + count;
%         assert(m_id <= DEF_MODEL_IDS.M_T3);
%         
%         [train_t_ids test_t_ids] = load_splits(source_domain, t_id, param);
%         d_tr_target = Data{t_id}(train_t_ids.target{i}, :);
%         l_tr_target = Labels{t_id}(train_t_ids.target{i});
%         data.train.targets = [data.train.targets; d_tr_target];
%         labels.train.targets = [labels.train.targets l_tr_target];
%         data.train.target_model_ids = [data.train.target_model_ids; m_id * ones(length(l_tr_target),1)];
%         
%         d_te_target = Data{t_id}(test_t_ids.target{i},:);
%         l_te_target = Labels{t_id}(test_t_ids.target{i});
%         data.test.targets = [data.test.targets; d_te_target];
%         labels.test.targets = [labels.test.targets l_te_target];
%         data.test.target_model_ids = [data.test.target_model_ids; m_id*ones(length(l_te_target),1)];
%     end
%     
%     % Domain classification
%     X_tr = data.train.targets;
%     C_tr = labels.train.targets;
%     Y_tr = data.train.target_model_ids - 1;
%     
%     X_te = data.test.targets;
%     C_te = labels.test.targets;
%     Y_te = data.test.target_model_ids - 1;
%     
%     param.num_clusters = 3;
%     % fprintf('Domain Clustering into %d Domains . . .\n', param.num_clusters);
%     Y_hat = DomainDiscovery (X_tr, C_tr, param.num_clusters, Y_tr, param);
%     acc_hfm(i) = ClusterAccuracy(Y_hat, Y_tr);
%     % fprintf('\n Split %d, domain Clustering accuracy %f . . .', i, bestAcc);
%     
%     opts = struct;
%     opts.depth = 3;
%     opts.numTrees = 100;
%     opts.numSplits = 5;
%     opts.numDomains = 3;
%     opts.verbose = false;
%     opts.param = param;
%     % train RF by pooling all data
% %     m = forestTrain(X_tr, Y_tr, opts, C_tr);
% %     Y_hat = forestTest(m, X_te);
% %     ac = sum(Y_hat == Y_te)/length(Y_te);
%     m = treeTrain_domainCluster(X_tr, [], opts, C_tr);
%     Y_hat = treeTest_domainCluster(m, X_tr);
%     ac = sum(Y_hat == Y_tr)/length(Y_tr);
%     
% %     model = classRF_train(dt_tr, lb_tr');
% %     Y_hat = classRF_predict(dt_te, model);
%     
%     num_class = 3;
%     numTest = zeros(num_class, 1);
%     numTest = vl_binsum(numTest, ones(length(Y_tr'),1), Y_tr');
%     idx = sub2ind([num_class, num_class], ...
%         Y_tr, Y_hat);
%     confus = zeros(num_class) ;
%     confus = vl_binsum(confus, ones(size(idx)), idx);
%     acc_our(i) = 100*mean(diag(confus)./numTest);
%     % fprintf('\n RF: Split %d, domain classification accuracy = %f', i, acc);
%     
%     
% %     param.weights = param.svm.C *ones(length(mid_tr), 1);
% %     svm = train(param.weights, ...
% %         mid_tr-1, ...
% %         sparse(dt_tr),  ...
% %         sprintf('-B %f -c %f -q', ...
% %         param.svm.biasMultiplier, param.svm.C)) ;
% %     w = svm.w' ;
% %     model.b = param.svm.biasMultiplier * w(end, :) ;
% %     model.w = w(1:end-1, :) ;
% %     model.svmmodel = svm;
% %     param.categories = 1:3;
% %     acc_src = test_svm(model, mid_te'-1, dt_te);
% %     fprintf('\n SVM: Split %d, domain classification accuracy = %f', i, acc_src);
%     % train RF per category
% %     for c=1:length(param.categories)
% %         index = (lb_tr == c);
% %         X1  = dt_tr(index,:);
% %         Y1  = mid_tr(index);
% %         % train
% %         m= forestTrain(X1, Y1, opts);
% %         % test
% %         index = (lb_te == c);
% %         X2  = dt_te(index,:);
% %         Y2  = mid_te(index);
% %         ypredict = forestTest(m, X2);
% %         
% %         ac = sum(ypredict == Y2)/length(Y2);
% %         
% %         fprintf('\n category %d, domain classification accuracy = %f', c, ac);
% %     end   
% end
% acc_hfm_avg = sum(acc_hfm)/n;
% acc_our_avg = sum(acc_our)/n;
% fprintf('\n Average acc-hfm %f, acc-our %f',  acc_hfm_avg, acc_our_avg);