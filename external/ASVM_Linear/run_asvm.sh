echo "########## Train source model ##########"
./train_asvmlinear data/heart_scale model_src
echo "########## Test source model ##########"
./predict data/heart_scale model_src out.txt
echo "########## Train adaptive model ###########"
./train_asvmlinear data/heart_scale + model_src model_tar
echo "########## Test adaptive model ###########"
./predict data/heart_scale model_tar out.txt
