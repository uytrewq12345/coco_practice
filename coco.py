# 初回のみ実行
! pip install pycocotools
! wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
! unzip annotations_trainval2014.zip


from pycocotools.coco import COCO
import pandas as pd
import tqdm

json_filename = "annotations/instances_val2014.json"
coco = COCO(json_filename)

def add_xmin_ymin_xmax_ymax(annotations):
    for i in range( len(annotations) ):
        annotations[i]["xmin"] = annotations[i]["bbox"][0]
        annotations[i]["ymin"] = annotations[i]["bbox"][1]
        annotations[i]["xmax"] = annotations[i]["bbox"][0] + annotations[i]["bbox"][2]
        annotations[i]["ymax"] = annotations[i]["bbox"][0] + annotations[i]["bbox"][3]
    return annotations

def calc_iou(a, b):
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou

def get_bbox(ann):
    return [  ann["xmin"],  ann["ymin"],  ann["xmax"],  ann["ymax"] ]


def detect_max_iou_bbox(annotations):
    for i in range( len(annotations) ):
        annotations[i]["max_iou"] = 0
        annotations[i]["max_iou_category_id"] = -1

    for i in range( len(annotations) ):
        for j in range(i + 1, len(annotations)):

            bbox_i = get_bbox(annotations[i])
            bbox_j = get_bbox(annotations[j])
            iou = calc_iou( bbox_i, bbox_j )


            if iou >  annotations[i]["max_iou"]:
                annotations[i]["max_iou"] = iou
                annotations[i]["max_iou_category_id"] = annotations[j]["category_id"]

            if iou >  annotations[j]["max_iou"]:
                annotations[j]["max_iou"] = iou
                annotations[j]["max_iou_category_id"] = annotations[i]["category_id"]

    return annotations


annotation_list = []

for image_id, image_info in tqdm.tqdm(coco.imgs.items()):
    filename = image_info["file_name"]
    ann_ids = coco.getAnnIds( imgIds = [image_id] )
    annotations = coco.loadAnns(ids = ann_ids)
    annotations = add_xmin_ymin_xmax_ymax(annotations)
    annotations = detect_max_iou_bbox(annotations)
    annotation_list += annotations


df = pd.DataFrame(annotation_list)

df

# 各クラスの単体：そうでない
iou_thresh = 0.5
df[ df["max_iou"] >= iou_thresh ]["max_iou_category_id"].value_counts()



# plot
import seaborn as sns
df["center_x"] = (df["xmax"] + df["xmin"]) / 2
df["center_y"] = (df["ymax"] + df["ymin"]) / 2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# plt.hist2d(df["center_x"], df["center_y"],   bins=[ [0, 100, 200, 300 ],  [0, 100, 200, 300 ]], cmap=cm.jet)
ret = plt.hist2d(df["center_x"], df["center_y"])
plt.colorbar(ret[3])