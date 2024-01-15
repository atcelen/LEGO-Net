import os, json
import numpy as np
from tqdm import tqdm

TDF_DATA_DIR = r"D:\Datasets\processed-bedroom-diningroom-library-livingroom"
SGF_DATA_DIR = r"D:\Datasets\SG_FRONT"

bedroom_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe", "start", "end"]
livingroom_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "start", "end"]
library_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "dressing_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "wardrobe", "wine_cabinet", "start", "end"]

cat2sup_cat = {
    "chair":"seating", 
    "bookshelf":"storage",
    "cabinet":"storage",
    "table":"surface",
    "desk":"surface",
    "lamp":"lighting",
    "bed":"seating",
    "shelf":"storage",
    "tv_stand":"surface",
    "sofa":"seating",
    "floor":None,
    "wardrobe":"storage",
    "nightstand":"surface",
}

#Load the mapping
with open(os.path.join(SGF_DATA_DIR, "mapping.json"), "r") as json_file:
    mapping = json.load(json_file)
# Load SGFront
json_files = [file for file in os.listdir(SGF_DATA_DIR) if file.endswith('.json') and 'relationships_all' in file]
all_scans = []
for file in json_files:
    with open(os.path.join(SGF_DATA_DIR, file), "r") as json_file:
        data = json.load(json_file)
    all_scans += [item for item in data["scans"]]

folder_names = [f for f in os.listdir(TDF_DATA_DIR) if f.startswith("processed") and f.endswith("augmented")]
for folder_name in folder_names:    
    if "bedroom" in folder_name:
        category_mapping = bedroom_idx
    elif "livingroom" in folder_name or "diningroom" in folder_name:
        category_mapping = livingroom_idx
    elif "library" in folder_name:
        category_mapping = library_idx
    
    scene_dir = os.path.join(TDF_DATA_DIR, folder_name)
    if os.path.exists(os.path.join(scene_dir, "data_tv_ctr.npz")):
        data_tv = np.load( os.path.join(scene_dir, "data_tv_ctr.npz"), allow_pickle=True)
        data_tv = dict(data_tv)
        # KEYS : ['scenedirs', 'nbj', 'pos', 'ang', 'siz', 'cla', 'vol', 'fpoc', 'nfpc', 'ctr', 'fpbpn']
    if os.path.exists(os.path.join(scene_dir, "data_test_ctr.npz")):
        data_test = np.load( os.path.join(scene_dir, "data_test_ctr.npz"), allow_pickle=True)
        data_test = dict(data_test)
        # KEYS : ['scenedirs', 'nbj', 'pos', 'ang', 'siz', 'cla', 'vol', 'fpoc', 'nfpc', 'ctr', 'fpbpn']

    data_tv["sg"] = []
    data_test["sg"] = []
    print(f"Processing {folder_name} for {len(data_tv['scenedirs'])} train and {len(data_test['scenedirs'])} test scenes")
    for i in tqdm(range(len(data_tv["scenedirs"]))):
        scene_id = data_tv["scenedirs"][i]
        scene_id = scene_id.split("_")[1]
        
        found_dict = [item for item in all_scans if item["scan"] == scene_id]
        if found_dict == []:
            print(scene_id)
            print("---------------")
            continue
        elif len(found_dict) > 1:
            print(scene_id)
            print(found_dict)
            print("---------------")
            break
        else:
            found_dict = found_dict[0]
            sg_scene_objs = list(found_dict["objects"].values())
            sg_scene_objs = [item for item in sg_scene_objs if item not in ["floor", "ceiling_lamp", "pendant_lamp"]]
            row_, col_ = np.nonzero(data_tv["cla"][i])
            atiss_objs = list(np.array(category_mapping)[col_])
            atiss_objs = [item for item in atiss_objs if item not in ["ceiling_lamp", "pendant_lamp"]]
            if sg_scene_objs != atiss_objs:
                print(scene_id)
                print("3D-Front from the download link")
                print(sg_scene_objs)
                print("3D-Front from the provided Google Drive")
                print(atiss_objs)
                print("-------------------")
            else:
                rel = [[
                    found_dict["objects"][str(item[0])],
                    found_dict["objects"][str(item[1])],
                    item[3]
                    ] for item in found_dict["relationships"]
                ]
                data_tv["sg"].append(rel)
    
    for i in tqdm(range(len(data_test["scenedirs"]))):
        scene_id = data_test["scenedirs"][i]
        scene_id = scene_id.split("_")[1]
        
        found_dict = [item for item in all_scans if item["scan"] == scene_id]
        if found_dict == []:
            print(scene_id)
            print("---------------")
            continue
        elif len(found_dict) > 1:
            print(scene_id)
            print(found_dict)
            print("---------------")
            break
        else:
            found_dict = found_dict[0]
            sg_scene_objs = list(found_dict["objects"].values())
            sg_scene_objs = [item for item in sg_scene_objs if item not in ["floor", "ceiling_lamp", "pendant_lamp"]]
            row_, col_ = np.nonzero(data_test["cla"][i])
            atiss_objs = list(np.array(category_mapping)[col_])
            atiss_objs = [item for item in atiss_objs if item not in ["ceiling_lamp", "pendant_lamp"]]
            if sg_scene_objs != atiss_objs:
                print(scene_id)
                print("3D-Front from the download link")
                print(sg_scene_objs)
                print("3D-Front from the provided Google Drive")
                print(atiss_objs)
                print("-------------------")
            else:
                rel = [[
                    found_dict["objects"][str(item[0])],
                    found_dict["objects"][str(item[1])],
                    item[3]
                    ] for item in found_dict["relationships"]
                ]
                data_test["sg"].append(rel)
    
    data_tv["sg"] = np.array(data_tv["sg"], dtype=object)
    data_test["sg"] = np.array(data_test["sg"], dtype=object)
    np.savez(os.path.join(scene_dir, "data_tv_ctr.npz"), **data_tv)
    np.savez(os.path.join(scene_dir, "data_test_ctr.npz"), **data_test)