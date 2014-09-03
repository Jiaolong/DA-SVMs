function [model ] = PMT_SVM(Y, X, C, wsource,GAMMA) 

% Y     : training labels
% X     : training features as rows of X
% C     : Error weight for SVM formulations
% wsource: Parameters of the source SVM (it is assumed to be l2 normalized)
% GAMMA : Transfer parameter 

% See: Y. Aytar and A. Zisserman, 
% Tabula Rasa: Model Transfer for Object Category Detection, ICCV'11


if nargin < 5
    GAMMA  = 1;
end    
   
M = length(wsource(:));
L = size(X,1);


paramCount = M+L+1;
A=zeros(L*2+1,paramCount); 

cnt = 0;

shift_w = 0;
shift_zeta = M;

% SVM constraints
A(cnt+1:cnt+L,shift_w+1:shift_w+M) = -X.*(ones(M,1)*Y')';    
A(cnt+1:cnt+L,end) = -Y;
for  i=1:L
    cnt = cnt +1;       
    A(cnt,shift_zeta+i)=-1;
end
b = [-ones(L,1)];    

for  i=1:L
    cnt = cnt +1;       
    A(cnt,shift_zeta+i)=-1;
end
b = [b;zeros(L,1)];

% half space constraint   ws'*w >= 0
cnt = cnt +1;       
A(cnt,shift_w+1:shift_w+M) = -wsource(:)';    
b = [b;0];

A=sparse(A);

% squared term coefficients
w=wsource(:);
P0 = eye(length(w))-(w*w')/(norm(w)^2);
H = zeros(paramCount,paramCount);
H(shift_w+1:shift_w+M,shift_w+1:shift_w+M) = eye(M) + GAMMA*P0;
H = sparse(H);

% term coefficients
f = [0*wsource(:);C*ones(L,1);0];

P = quadprog(2*H,f,A,b,[],[],[],[],[],optimset('MaxIter',800));


model.b = P(end); 
model.w = P(shift_w+1:shift_w+M); 
model.tr_w = wsource(:);
model.C = C;
model.GAMMA = GAMMA;








