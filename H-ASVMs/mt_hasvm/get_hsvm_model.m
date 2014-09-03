function model = get_hsvm_model( model_id, models )
% model = get_hsvm_model( model_id, models )

for i=1:length(models)
    if model_id == models{i}.id
        model = models{i};
        return;
    end
end

