function exists_or_mkdir(dir_make)
if ~exist(dir_make, 'dir')
    mkdir(dir_make);
end
end