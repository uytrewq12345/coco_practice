import json
import tqdm
import pandas as pd

def read_json(json_filepath):
    fp = open(json_filepath, 'r')
    json_data = json.load(fp)
    fp.close()

    return json_data

def update_image_id(coco_json, base_id):
    for i  in range( len(coco_json["images"]) ) :
        coco_json["images"][i]["id"] += base_id

    for i in range( len(coco_json["annotations"]) ) :
        coco_json["annotations"][i]["image_id"] += base_id

    return coco_json

def update_annotation_id(coco_json, base_id):

    for i in range( len(coco_json["annotations"]) ) :
        coco_json["annotations"][i]["id"] += base_id

    return coco_json


def get_max_image_id(coco_json):
    max_id = 0
    for img in coco_json["images"]:
        max_id = max(max_id, img["id"])
    return max_id

def get_max_annotation_id(coco_json):
    max_id = 0
    for ann in coco_json["annotations"] :
        max_id = max( max_id,  ann["id"] )
    return max_id

def delete_duplicate_category_id( categories ):
    merged_categories = []
    id_set = set()
    for category_dict in categories:
        if not category_dict["id"] in id_set:
            id_set.add( category_dict["id"] )
            merged_categories.append( category_dict )


    return merged_categories


def merge_coco_json(coco_filelist, keys):

    merged_coco_dict = {}
    for key in keys:
        merged_coco_dict[key] = []


    for coco_file in tqdm.tqdm(coco_filelist):

        coco_dict  = read_json(coco_file)
        coco_dict  = update_image_id(coco_dict,  get_max_image_id(merged_coco_dict) )
        coco_dict  = update_annotation_id(coco_dict,  get_max_annotation_id(merged_coco_dict) )

        for key in keys:
            merged_coco_dict[key] += coco_dict[key]

    merged_coco_dict["categories"] = delete_duplicate_category_id(merged_coco_dict["categories"])

    return merged_coco_dict


# main
coco_filelist = [
    "annotations/instances_val2014.json",
    "annotations/instances_val2017.json",
]

keys = {
    "info"   ,
    "images" ,
    "licenses", 
    "annotations" ,
    "categories"  ,
}

coco_json = merge_coco_json( coco_filelist = coco_filelist, keys = keys )

pd.DataFrame(coco_json["images"])["id"].value_counts()

pd.DataFrame(coco_json["annotations"])["id"].value_counts()