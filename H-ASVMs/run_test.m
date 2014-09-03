% SSVM for Domain Adaptation
% Jiaolong Xu
% jiaolong@cvc.uab.es

clear all;
startup;

% Test various baselines:
% SRC,  TAR, SRC(SSVM), TAR(SSVM), ASVM, PMT-SVM
%test_svms;

% Test SSVM
% SRC(SSVM), ASSVM, MIX, COSS-SSVM
test_ssvms;

% Test H-ASVM with original domains
% test_hasvm;