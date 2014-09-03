function write_latex_tables
% Write the results into Latex table
% Load the results

cache_dir = './cache/';
m_fs = dir([cache_dir 'test_*.mat']);
rows = cell(length(m_fs), 1);
for i=1:length(m_fs)
    clear row;
    load([cache_dir m_fs(i).name]);
    rows{i} = row;
end

index_colums_display = 13;
table_normal(index_colums_display, rows, cache_dir, 'table3.tex');
end

function table_normal(index_colums_display, rows, cache_dir, filename)
str_columns = 'Method ';
str_rr = '|r|';
if ~isempty(rows) && ~isempty(rows{1})
    items = rows{1}.items;
    for i=index_colums_display
        str_c = strrep(items{i}.name, '->', '$\\to$');
        str_columns = [str_columns ' & ' str_c];
        str_rr = [str_rr 'r|'];
    end
end

FID = fopen([cache_dir filename], 'w');
% FID = 1;
fprintf(FID, ['\\begin{tabular}{' str_rr '}\\hline \n']);
fprintf(FID, [str_columns '\\\\ \\hline \n']);
for r=1:length(rows)
    row = rows{r};
    str_row = row.name;
    items = rows{r}.items;
    for i=index_colums_display
        str_row = [str_row sprintf(' & %.1f', items{i}.value)];
        if isfield(items{i}, 'err')
            str_row = [str_row ' {$\\pm$} ' sprintf('%.1f', items{i}.err)];
        end
    end
    str_row = [str_row '\\\\ \\hline \n'];
    fprintf(FID, str_row);
end
 
fprintf(FID, '\\end{tabular}\n');
fclose(FID);
end

function table_transpose(index_colums_display)
end