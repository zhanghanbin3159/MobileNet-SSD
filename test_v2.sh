#!/bin/sh
latest=snapshot_v2/0921_mobilenet_iter_20000.caffemodel
#latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
/media/ziwei/Harddisk02/HanBin/TOOL/workspace_caffe/ssd/build/tools/caffe train -solver="solver_test.prototxt" \
-weights=$latest \
-gpu 0
