#!/bin/sh
if ! test -f example/V2_4category/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot2
/home/pesong/tools/ssd-caffe/build/tools/caffe train -solver="solver_train.prototxt" \
-weights="mobilenetv2_iter_300000.caffemodel" \
-gpu 0 
