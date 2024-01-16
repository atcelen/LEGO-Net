import os, json
import numpy as np
from tqdm import tqdm
from copy import deepcopy

TDF_DATA_DIR = r"D:\Datasets\processed-bedroom-diningroom-library-livingroom"
SGF_DATA_DIR = r"D:\Datasets\SG_FRONT"

bedroom_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe", "start", "end"]
livingroom_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "start", "end"]
library_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "dressing_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "wardrobe", "wine_cabinet", "start", "end"]

MAX_N_OBJ = 21
MAX_N_EDGES = 275
MAX_FPOC = 51
N_NODE_TYPES = 5
N_EDGE_TYPES = 5 


# Create the super-category to index mapping dictionary
sup_cat2idx = {
    "seating" : 0,
    "storage" : 1,
    "surface" : 2,
    "lighting" : 3,
    "decor" : 4
}

# Create the class to super-category mapping dictionary
cla2sup_cat = {
    'console_table' : "surface",
    'children_cabinet' : "storage",
    'stool' : "seating",
    'wine_cabinet' : "storage",
    'corner_side_table' : "surface",
    'dressing_table' : "surface",
    'dressing_chair' : "seating",
    'armchair' : "seating",
    'chinese_chair' : "seating",
    'tv_stand' : "surface",
    'kids_bed' : "seating",
    'shelf' : "storage",
    'table' : "surface",
    'ceiling_lamp' : "lighting",
    'chair' : "seating",
    'desk' : "surface",
    'lazy_sofa' : "seating",
    'nightstand' : "surface",
    'bookshelf' : "storage",
    'wardrobe' : "storage",
    'dining_chair' : "seating",
    'l_shaped_sofa' : "seating",
    'lounge_chair' : "seating",
    'chaise_longue_sofa' : "seating",
    'sofa' : "seating",
    'cabinet' : "storage",
    'loveseat_sofa' : "seating",
    'coffee_table' : "surface",
    'double_bed' : "seating",
    'pendant_lamp' : "lighting",
    'single_bed' : "seating",
    'multi_seat_sofa' : "seating",
    'dining_table' : "surface",
    'round_end_table' : "surface",
} 
# Create the preposition to index mapping dictionary
prep2idx = {
    "left" : 0,
    "right" : 1,
    "front" : 2,
    "behind" : 3,
    "close by" : 4 
}

def preprocess(data):
    new_classes_array = np.zeros((data["cla"].shape[0], data["cla"].shape[1], 5))
    for i in tqdm(range(len(data["scenedirs"]))):
        scene_id = data["scenedirs"][i]
        scene_id = scene_id.split("_")[1]
        
        found_items = [item for item in all_scans if item["scan"] == scene_id]
        if found_items == []:
            print(scene_id)
            print("---------------")
            continue
        elif len(found_items) > 1:
            print(scene_id)
            print(found_items)
            print("---------------")
            break
        else:
            found_dict = deepcopy(found_items[0])
            # sg_scene_objs = list(found_dict["objects"].values())
            # sg_scene_objs = [item for item in sg_scene_objs if item not in ["floor", "ceiling_lamp", "pendant_lamp"]]
            # row_, col_ = np.nonzero(data["cla"][i])
            # atiss_objs = list(np.array(category_mapping)[col_])
            # atiss_objs = [item for item in atiss_objs if item not in ["ceiling_lamp", "pendant_lamp"]]
            # if sg_scene_objs != atiss_objs:
                # print(scene_id)
                # print("3D-Front from the download link")
                # print(sg_scene_objs)
                # print("3D-Front from the provided Google Drive")
                # print(atiss_objs)
                # print("-------------------")
                # raise ValueError("The objects in the downloaded 3D-Front data and the provided Google Drive data do not match")
            # else:
            # Delete floor and light relationships
            items_to_delete = []
            for item in found_dict["relationships"]:
                if any(found_dict["objects"][str(item[i])] in ["floor", "ceiling_lamp", "pendant_lamp"] for i in range(2)) or item[2] > 5:
                    items_to_delete.append(item)
            for item in items_to_delete:
                found_dict["relationships"].remove(item)
            # Delete floor and lights
            keys_to_delete = []
            for key, val in found_dict["objects"].items():
                if val not in ["floor", "ceiling_lamp", "pendant_lamp"]:
                    found_dict[key] = cla2sup_cat[val]
                else:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del found_dict["objects"][key]
            # Remap objects to super-categories
            for key, val in found_dict["objects"].items():
                found_dict["objects"][key] = cla2sup_cat[val]
            edge_one_hot = np.zeros((MAX_N_EDGES, 2 * MAX_N_OBJ + N_EDGE_TYPES))
            for idx, item in enumerate(found_dict["relationships"]):
                edge_one_hot[idx, item[0] - 1] = 1
                edge_one_hot[idx, prep2idx[item[3]]+ N_NODE_TYPES] = 1
                edge_one_hot[idx, item[1] - 1 + 2 * N_NODE_TYPES] = 1
            data["sg"].append(edge_one_hot.tolist())
            # Change the class to super-category
            # for new_cl, atiss_obj_row in zip(new_classes_array[i], map(cla2sup_cat.get, atiss_objs)):
            #     new_cl[sup_cat2idx[atiss_obj_row]] = 1

    # data["cla"] = new_classes_array
    data["sg"] = np.array(data["sg"])
    return data

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
    data_tv = preprocess(data_tv)
    data_test = preprocess(data_test)

    # Pad the data
    for key in ["pos", "ang", "siz", "vol", "cla"]:
        existing_array = data_tv[key]
        desired_shape = (existing_array.shape[0], MAX_N_OBJ, existing_array.shape[2])
        padding = [(0, max(0, desired_shape[i] - existing_array.shape[i])) for i in range(3)]
        padded_array = np.pad(existing_array, padding, mode='constant', constant_values=0)
        data_tv[key] = padded_array
    for key in ["pos", "ang", "siz", "vol", "cla"]:
        existing_array = data_test[key]
        desired_shape = (existing_array.shape[0], MAX_N_OBJ, existing_array.shape[2])
        padding = [(0, max(0, desired_shape[i] - existing_array.shape[i])) for i in range(3)]
        padded_array = np.pad(existing_array, padding, mode='constant', constant_values=0)
        data_test[key] = padded_array    
    for key in ["fpoc"]:
        existing_array = data_tv[key]
        desired_shape = (existing_array.shape[0], MAX_FPOC, existing_array.shape[2])
        padding = [(0, max(0, desired_shape[i] - existing_array.shape[i])) for i in range(3)]
        padded_array = np.pad(existing_array, padding, mode='constant', constant_values=0)
        data_tv[key] = padded_array

        existing_array = data_test[key]
        desired_shape = (existing_array.shape[0], MAX_FPOC, existing_array.shape[2])
        padding = [(0, max(0, desired_shape[i] - existing_array.shape[i])) for i in range(3)]
        padded_array = np.pad(existing_array, padding, mode='constant', constant_values=0)
        data_test[key] = padded_array 
    
    
    np.savez(os.path.join(scene_dir, "data_tv_ctr.npz"), **data_tv)
    np.savez(os.path.join(scene_dir, "data_test_ctr.npz"), **data_test)