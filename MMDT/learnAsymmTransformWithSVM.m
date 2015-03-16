% L = learnAsymmTransformWithSVM(XA, yA, XB, yB, params)
%
% XA, yA: examples and labels in source domain
% XB, yB: examples and labels in target domain
% params.gamma:  gamma parameter 
% params.use_Gaussian_kernel: 0 (linear) or 1 (Gaussian)
%
% Returns: 
% L: Transformation matrix
%
% Author Judy Hoffman (jhoffman@eecs.berkeley.edu)

function L = learnAsymmTransformWithSVM(XA, yA, XB, yB, params)
dA = size(XA,2);
dB = size(XB,2);

C = GetConstraints(yA,yB);

if dA ~= dB
    K0aa = formKernel(XA, XA, params);
    K0bb = formKernel(XB, XB, params);
    
    C(:,2) = C(:,2) - length(yA);
    S = AsymmetricFrob_slack_kernel2(K0aa,K0bb,C,params.gamma,10e-3);
    params.S = S;
    
    L = eye(dA, dB) + XA' * S * XB;
else
    X = [XA; XB];
    
    K0train = formKernel(X, X, params);
    S = asymmetricFrob_slack_kernel(K0train,C,params.gamma,10e-3);
    
    L = eye(dA) + X' * S * X;
end
end

function C = GetConstraints(y1, y2)
pos=1;

ly1=length(y1);
ly2=length(y2);
C=zeros(ly1*ly2,4);
for i=1:ly1
    for j=1:ly2
        if(y1(i)==y2(j))
            % w'Ax > 1; -w'Ax < -1;
            C(pos,:)=[i j+ly1 1 -1];
        else
            % w'Ax < -1
            C(pos,:)=[i j+ly1 -1 1];
        end
        pos=pos+1;
    end
end
end
