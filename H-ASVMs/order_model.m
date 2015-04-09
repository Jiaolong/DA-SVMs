function model = order_model( model )
% Adjust the order of learned weights
% 
model_copy = model;
for i=1:length(model_copy.Label)
    ind = model_copy.Label(i);
    model.w(:, ind) = model_copy.w(:, i);
    model.b(ind) = model_copy.b(i);
    model.Label(i) = i;
end
end

