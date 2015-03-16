DA-SVMs
=======

Various SVM based methods for domain adaptation.
###For object recognition experiments
```sh
$ ./download_data.sh
```
###For scene segmentation experiments:
```sh
$ mkdir data
$ cd data
$ ln -s <YourPath>/LauraVid .
$ ln -s <YourPath>/CamVid .
$ run da_scene.m
```

If you find the code useful, please consider to cite the following paper:

- J. Xu, S. Ramos, D. Vazquez, A. M. Lopez.
Cost-sensitive Structured SVM for Multi-category Domain Adaptation.
In International Conference on Pattern Recognition (ICPR), 2014.

- J. Xu, S. Ramos, D. Vazquez, A. M. Lopez.
Hierarchical Adaptive Structural SVM for Domain Adaptation.
In arXiv:1408.5400, 2014
