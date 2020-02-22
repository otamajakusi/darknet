import pickle
import os
import sys
import glob
import csv
import cv2
from tqdm import tqdm
import joblib
import random

classes = ['bk',
           'm1','m2','m3','m4','m5','m6','m7','m8','m9',
           'p1','p2','p3','p4','p5','p6','p7','p8','p9',
           's1','s2','s3','s4','s5','s6','s7','s8','s9',
           'sc','sh','sw','wn','wp','ws','wt']

def get_size(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    return h, w

def convert(img_name, box):
    _h, _w = get_size(img_name)
    dw = 1./(_w)
    dh = 1./(_h)
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def usage(cmd):
    print("Usage:", cmd, "data_dir anno_dir [single]")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage(sys.argv[0])
        sys.exit(1)
    cwd = os.getcwd()
    data_dir = sys.argv[1]
    anno_dir = sys.argv[2]
    single = True if len(sys.argv) == 4 else False
    labels_dir = 'labels'
    print('data_dir={}, anno_dir={}, single={}'.format(data_dir, anno_dir, single))
    with open('train.txt', 'w') as train, open('valid.txt', 'w') as valid:
        annos = sorted(glob.glob(os.path.join(anno_dir, '*.csv')))
        #print(annos, anno_dir)
        for anno in tqdm(annos):
            img = os.path.splitext(os.path.basename(anno))[0]
            img_name = os.path.join(data_dir, img)
            img_id = os.path.join(data_dir, os.path.splitext(img)[0]) + '.txt'
            #print(img, img_name, img_id)
            # debug
            #if not img.startswith('DSC07267.JPG'):
            #    continue
            if not os.path.exists(img_name):
              continue
            has_anno = False
            with open(img_id, 'w') as txt:
                with open(anno, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        cls_id = 0 if single else classes.index(row[0])
                        b = (float(row[1]), float(row[3]), float(row[2]), float(row[4]))
                        bb = convert(img_name, b)
                        #print('.', end='', flush=True)
                        has_anno = True
                        txt.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            if has_anno:
                if random.randint(0, 4) == 0:
                    valid.write('{}\n'.format(os.path.join(cwd, img_name)))
                else:
                    train.write('{}\n'.format(os.path.join(cwd, img_name)))

