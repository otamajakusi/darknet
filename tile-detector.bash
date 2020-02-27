#!/bin/bash

# git clone https://github.com/otamajakusi/darknet.git
# cd darknet

# before running this script, google drive should be mounted.
# tile score w/ darknet
googledrive="/content/drive/My\ Drive"
stat "${googledrive}"
if [ $? -ne 0 ]; then
    echo "google drive should be mounted"
    exit 1
fi

make

# tar -C directory --strip-components 1 has a problem for darknet??
# `'Couldn't open file:`
tar zxf /content/drive/My\ Drive/public/2018-06-23.tgz
tar zxf /content/drive/My\ Drive/public/2018-09-15.tgz
tar zxf /content/drive/My\ Drive/public/2018-09-29.tgz
tar zxf /content/drive/My\ Drive/public/2018-12-20.tgz
mv 2018-09-15/* 2018-09-29/* 2018-12-20/* 2018-06-23
mkdir anno
tar zxf /content/drive/My\ Drive/public/tile-labels.tgz -C anno

if [ "/content/drive/My\ Drive/ml/train.txt" -a "/content/drive/My\ Drive/ml/valid.txt" ]; then
    cp "/content/drive/My\ Drive/ml/{train,valid}.txt" .
else
    python3 scripts/tile_label.py 2018-06-23 anno
fi

weight="darknet53.conv.74"
if [ -d "/content/drive/My\ Drive/ml/yolo3-backup/yolov3-tile.backup" ]; then
    weight=backup/yolov3-tile.backup
elif [ -f "/content/drive/My\ Drive/ml/darknet53.conv.74" ]; then
    cp "/content/drive/My\ Drive/ml/darknet53.conv.74" .
else
    wget https://pjreddie.com/media/files/darknet53.conv.74
fi
rm -rf backup
ln -s "/content/drive/My\ Drive/ml/yolo3-backup" backup

./darknet detector train cfg/tiles.data cfg/yolov3-tile.cfg ${backup}
