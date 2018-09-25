#!/bin/sh
if ! test -f example/V2_4category/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot_v2
/media/ziwei/Harddisk02/HanBin/TOOL/workspace_caffe/ssd/build/tools/caffe train -solver="solver_train.prototxt" \
-snapshot="snapshot_v2/0921_mobilenet_iter_846.solverstate" \
-gpu 0
