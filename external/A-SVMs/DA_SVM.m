function [model wt] = DA_SVM(Y, X, C, wsource,GAMMA,LAMBDA,Neighbor) 

% Y     : training labels
% X     : training features as rows of X
% C     : Error weight for SVM formulations
% wsource: Parameters of the source SVM (it is assumed to be l2 normalized)
% GAMMA : Transfer parameter 
% LAMBDA: The weight of the deformation
% Neighbor: Deformation neighborhood of each cell  

% See: Y. Aytar and A. Zisserman, 
% Tabula Rasa: Model Transfer for Object Category Detection, ICCV'11


if nargin < 5
    GAMMA = 0.01;
end    
    
if nargin < 6
    LAMBDA = 0.0001;
end

if nargin < 7
    Neighbor = 4; % N Neighborhood
end



Height=size(wsource,1);
Width=size(wsource,2);
M = size(wsource,1)*size(wsource,2);
D = size(wsource,3);
L = size(X,1);

ws = reshape(wsource,M,size(wsource,3)); % (1,1)   (2,1)   (3,1)   (4,1) ... (2,1)  (2,2)

INF_VAL = 1000000;

% easier reach for spatial positions of cells
pos = [];
k=0;
ind=[];
for j=1:size(wsource,2)
    for i=1:size(wsource,1)
        k=k+1;
        pos(k,:) = [i j];
        ind(i,j) = k;
    end
end

paramCount = M*M+M*D+L+1;
A=zeros(L+L,paramCount);

cnt = 0;

% define distance panalizations
Dist = INF_VAL * ones(M*M,1);
for k=1:M    
    y = pos(k,1); 
    x = pos(k,2);            
    for i=max(1,y-Neighbor):min(Height,y+Neighbor)
        for j=max(1,x-Neighbor):min(Width,x+Neighbor)                        
            Dist((k-1)*M+ind(i,j)) = min(INF_VAL,max(0.5,norm(pos(k,:) - [i j])));
        end
    end
end

shift_flow = 0;
shift_w = M*M;
shift_zeta = M*M+M*D;


% SVM constraints
A(cnt+1:cnt+L,shift_w+1:shift_w+M*D) = -X.*(ones(M*D,1)*Y')';    
A(cnt+1:cnt+L,end) = -Y;
for  i=1:L
    cnt = cnt +1;       
    A(cnt,shift_flow+1:shift_flow+M*M) =GAMMA*reshape(-(reshape(X(i,:),M,D)*ws')'*Y(i),M*M,1);
    A(cnt,shift_zeta+i)=-1;
end
b = -ones(L,1);    

for  i=1:L
    cnt = cnt +1;       
    A(cnt,shift_zeta+i)=-1;
end
b = [b;zeros(L,1)];    

A=sparse(A);

% squared term coefficients
H = sparse(paramCount,paramCount);
for i=1:M*D
    H(shift_w+i,shift_w+i) = 1;
end

for i=1:M*M
    H(shift_flow+i,shift_flow+i) = LAMBDA*Dist(i);
end

DistDiag=reshape((-2*eye(M).*reshape(Dist,M,M)),M*M,1);
 
% term coefficients
f = [DistDiag*LAMBDA;zeros(M*D,1);C*ones(L,1);0];

P = quadprog(2*H,f,A,b);

F = reshape(P(1:M*M),M,M);
    wt = ws*0;
    for i=1:M
        wt(i,:) = F(:,i)' * ws ;
    end

model.b = P(end); 
model.w = P(shift_w+1:shift_w+M*D)+GAMMA*wt(:); 
model.tr_w = ws;
model.C = C;







