function [S slack i] = AsymmetricFrob_slack_kernel2(KA,KB,C,gamma,thresh)
%Frobenius-based transformation learning
if (~exist('thresh')),
    thresh=10e-3;
end
if (~exist('gamma')),
    gamma = 1e1;
end
maxit = 1e6;
[nA,nA] = size(KA);
[nB,nB] = size(KB);
S = zeros(nA,nB);
[c,t] = size(C);
slack = zeros(c,1);
lambda = zeros(c,1);
lambda2 = zeros(c,1);
%v = (C(:,1)-1)*n+C(:,2);
%viol = C(:,4).*(K0(v)-C(:,3));
viol = -1*C(:,4).*C(:,3);
viol = viol';

for i = 1:maxit
    [mx,curri] = max(viol);
    if mod(i,1000) == 0
        %fprintf(1,'Iteration %d, maxviol %d\n', i, mx);
    end
    if mx < thresh
        break;
    end
    p1 = C(curri,1);
    p2 = C(curri,2);
    b = C(curri,3);
    s = C(curri,4);
    kx = KA(p1,:);
    ky = KB(:,p2);
    
    alpha = min(lambda(curri),(s*(b-kx*S*ky)-slack(curri)) / (1/gamma + KA(p1,p1)*KB(p2,p2)));
    lambda(curri) = lambda(curri) - alpha;
    S(p1,p2) = S(p1,p2) + s*alpha;
    slack(curri) = slack(curri) - alpha/gamma;
    alpha2 =  min(lambda2(curri),gamma*slack(curri));
    slack(curri) = slack(curri) - alpha2/gamma;
    lambda2(curri) = lambda2(curri) - alpha2;
        
    %update viols
    %v = KA(C(:,1),p1);
    %w = KB(p2,C(:,2))';
    viol = viol + s*alpha*C(:,4)'.*KA(C(:,1),p1)'.*KB(p2,C(:,2));
    viol(curri) = viol(curri) + (alpha+alpha2)/gamma;
end

