#!/usr/bin/env sh

<PATH_TO_CAFFE>/build/tools/caffe train \
    --solver=<PATH_TO_FOLDER>/solver.prototxt --weights=<PATH_TO_STORED_MODELS>/VGG16_fconv.caffemodel
