function [local_obj, global_obj, obj] = ComputeObjective(X, Z_l, mu, Z_g, m, c_g, c_l)
I = size(X,1);
J = size(mu,1);
K = size(m,1);
obj= 0;
local_obj = 0;
global_obj = 0;
for j = 1:J
    for i = 1:I
       if Z_l(i, j)==1
           local_obj = local_obj + norm(X(i,:) - mu(j,:))^2;
       end
    end
    for k = 1:K
        if Z_g(j, k)==1
            global_obj = global_obj + norm(mu(j,:)-m(k,:))^2;
        end
    end   
end
obj = c_g * global_obj + c_l * local_obj;
end