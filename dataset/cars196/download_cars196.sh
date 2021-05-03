cd data/cars196
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz

wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
# wget http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz

tar -xzvf cars_train.tgz 
tar -xzvf cars_test.tgz
tar -xzvf car_devkit.tgz