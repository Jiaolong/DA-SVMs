#!/bin/bash
# for using multiple cores, add to the above line #$ -pe mpich 8
echo "QSUB begin"

set mfile_run = "run_test_qsub.m"
set dir_root = "/home/cvc/jiaolong/Code_Git/DA-SVMs/"
set matlab_exc = /opt/matlab/bin/matlab

cd $dir_root

$matlab_exc -nojvm -nodisplay -r "run "$dir_root$mfile_run

echo "QSUB end"
