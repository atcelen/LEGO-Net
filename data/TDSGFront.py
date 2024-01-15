import torch
import csv, os, json
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv
from tqdm import tqdm

from utils import *
from distance import *

from filepath import *

TDF_DATA_DIR = r"D:\Datasets\processed-bedroom-diningroom-library-livingroom"
SGF_DATA_DIR = r"D:\Datasets\SG_FRONT"

cla2sup_cat = {
    "armchair" : "seating", 
    "bookshelf" : "storage", 
    "cabinet" : "storage", 
    "chaise_longue_sofa" : "seating", 
    "chinese_chair" : "seating", 
    "coffee_table" : "surface", 
    "console_table" : "surface", 
    "corner_side_table" : "surface", 
    "desk" : "surface", 
    "dining_chair" : "seating", 
    "dining_table" : "surface", 
    "l_shaped_sofa" : "seating", 
    "lazy_sofa" : "seating", 
    "lounge_chair" : "seating", 
    "loveseat_sofa" : "seating", 
    "multi_seat_sofa" : "seating", 
    "round_end_table" : "surface", 
    "shelf" : "storage", 
    "stool" : "seating", 
    "tv_stand" : "surface", 
    "wardrobe" : "storage", 
    "wine_cabinet" : "storage"
}

class TDSGFront():
    def __init__(self, use_augment=True, print_info=True):
        # train-test-val splits (70-20-10)
        split2room = {key : [] for key in ["train", "test", "val"]} # { "test": ['MasterBedroom-147840', 'MasterBedroom-49298', ...] }
        csv_files = [file for file in os.listdir(TDF_DATA_DIR) if file.endswith('.csv')]
        json_files = [file for file in os.listdir(SGF_DATA_DIR) if file.endswith('.json') and 'relationships_all' in file]
        # Read SG-Front JSON files
        all_scans = []
        for file in json_files:
            with open(os.path.join(SGF_DATA_DIR, file), "r") as json_file:
                data = json.load(json_file)
            all_scans += [item for item in data["scans"]]
        self.n_edges = 5 # left, right, front, behind, close by
        self.n_nodes = 4 # Storage Furniture, Seating Furniture, Surface Furniture, Lighting
        # Read 3D-FRONT csv files
        for file in csv_files:
            with open(os.path.join(TDF_DATA_DIR, file), "r") as csv_file:
                data = [row for row in csv.reader(csv_file, delimiter=',')]  # 6286 in list, 5668tv/224test for bedroom
            for s in ["train", "test", "val"]: 
                split_rooms = [room[0] for room in data if room[1] == s and room[0] not in split2room["train"] + split2room["test"] + split2room["val"]]
                split2room[s] += split_rooms
        # Remove rooms that are not in SG-Front
        all_scan_ids = [scan["scan"] for scan in all_scans]
        for s in ["train", "test", "val"]:
            split2room[s] = [room for room in split2room[s] if room in all_scan_ids]
        
        if use_augment:
            folder_names = [f for f in os.listdir(TDF_DATA_DIR) if f.startswith("processed") and f.endswith("augmented")]
        else:
            folder_names = [f for f in os.listdir(TDF_DATA_DIR) if f.startswith("processed") and not f.endswith("augmented")]
        
        self.scenes_tv = []
        self.scenes_test = []
        for folder_name in folder_names:
            self.scene_dir = os.path.join(TDF_DATA_DIR, folder_name)
            trainvalrooms = split2room["train"] + split2room["val"]
            self.scenes_tv   +=  [os.path.join(self.scene_dir, e) 
                                for e in list(os.listdir(self.scene_dir)) 
                                if e.split("_")[1] in trainvalrooms]
            # NOTE: selecting the 2nd argument works for augmented as well
            # Example: <scene_dir>/ff7b42d9-0e58-4847-8d47-f793a11cd3bd_MasterBedroom-83934(_0)
            self.scenes_tv = sorted(self.scenes_tv) 
            self.scenes_test+=  [ os.path.join(self.scene_dir, e) 
                                for e in list(os.listdir(self.scene_dir)) 
                                if e.split("_")[1] in split2room["test"]]
            self.scenes_test = sorted(self.scenes_test)
        
        if print_info: print(f"TDFDataset: len(self.scenes_tv)={len(self.scenes_tv)}, len(self.scenes_test)={len(self.scenes_test)}\n")

        # preload data into RAM
        if os.path.exists(os.path.join(self.scene_dir, "data_tv_ctr.npz")):
            self.data_tv = np.load( os.path.join(self.scene_dir, "data_tv_ctr.npz"), allow_pickle=True)
            # KEYS : ['scenedirs', 'nbj', 'pos', 'ang', 'siz', 'cla', 'vol', 'fpoc', 'nfpc', 'ctr', 'fpbpn']
        if os.path.exists(os.path.join(self.scene_dir, "data_test_ctr.npz")):
            self.data_test = np.load( os.path.join(self.scene_dir, "data_test_ctr.npz"), allow_pickle=True)
            # KEYS : ['scenedirs', 'nbj', 'pos', 'ang', 'siz', 'cla', 'vol', 'fpoc', 'nfpc', 'ctr', 'fpbpn']
        with open(os.path.join(self.scene_dir, "dataset_stats.txt")) as f:
            # Same regardless of if only living room (ctr.npz processed from boxes.npz, generated in one go from ATISS for all living+livingdiningrooms)
            # NOTE: data prepared with splits test+train+val (ATISS's preprocess_data.py)
            ds_js_all= json.loads(f.read()) 

        self.object_types = ds_js_all["object_types"] # class labels in order from 0 to self.cla_dim
        self.mapping = cla2sup_cat # mapping from class labels to super-categories
        self.maxnobj = 21 # based on ATISS Suplementary Material
        self.maxnfpoc = 51 # bedroom: 25 , livingroom: 51(based on preprocessing data)
        self.nfpbpn = 250
        
        self.pos_dim = 2 # coord in x, y (disregard z); normalized to [-1,1]
        self.ang_dim = 2 # cos(theta), sin(theta), where theta is in [-pi, pi]
        self.siz_dim = 2 # length of bounding box in x, y; normalized to [-1, 1]
        self.cla_dim = len(set(self.mapping.values())) # number of classes (19 for bedroom, 22 for all else)
        self.sha_dim = self.siz_dim+self.cla_dim

        self.cla_colors = list(plt.cm.rainbow(np.linspace(0, 1, self.cla_dim)))

        self.room_size = [12, 4, 12] #[rs_x, rs_y, rs_z]
    
    ## HELPER FUNCTION: agnostic of specific dataset configs
    @staticmethod
    def parse_cla(cla):
        """ cla: [nobj, cla_dim]

            nobj: scalar, number of objects in the scene
            cla_idx: [nobj,], each object's class type index. 
        """ 
        nobj = cla.shape[0]
        for o_i in range(cla.shape[0]):
            if np.sum(cla[o_i]) == 0: 
                nobj = o_i
                break
        cla_idx = np.argmax(cla[:nobj,:], axis=1) #[nobj,cla_dim] -> [nobj,] (each obj's class index)
        return nobj, cla_idx
    
    @staticmethod
    def reset_padding(nobjs, toreset):
        """ nobjs: [batch_size]
            toreset(2): [batch_size, maxnumobj, 2]
        """
        for scene_idx in range(toreset.shape[0]):
            toreset[scene_idx, nobjs[scene_idx]:,:]=0
        return toreset
    
    @staticmethod
    def get_objbbox_corneredge(pos, ang_rad, siz):
        """ pos: [pos_dim,]
            ang_rad: [1,1], rotation from (1,0) in radians
            siz: [siz_dim,], full bbox length

            corners: corner points (4x2 numpy array) of the rotated bounding box centered at pos and with bbox len siz,
            bboxedge: array of 2-tuples of numpy arrays [x,y]
        """
        siz = (siz.astype(np.float32))/2
        corners = np.array([[pos[0]-siz[0], pos[1]+siz[1]],  # top left (origin: bottom left)
                            [pos[0]+siz[0], pos[1]+siz[1]],  # top right
                            [pos[0]-siz[0], pos[1]-siz[1]],  # bottom left 
                            [pos[0]+siz[0], pos[1]-siz[1]]]) # bottom right #(4, 2)
        corners =  np_rotate_center(corners, np.repeat((ang_rad), repeats=4, axis=0), pos) # (4, 2/1/2) , # +np.pi/2, because our 0 degree means 90
            # NOTE: no need to add pi/2: obj already starts facing pos y, we directly rotate from that
        bboxedge = [(corners[2], corners[0]), (corners[0], corners[1]), (corners[1], corners[3]), (corners[3], corners[2])]
                    # starting from bottom left corner
        return corners, bboxedge
    
    @staticmethod
    def get_xyminmax(ptxy):
        """ptxy: [numpt, 2] numpy array"""
        return np.amin(ptxy[:,0]), np.amax(ptxy[:,0]), np.amin(ptxy[:,1]), np.amax(ptxy[:,1])
    
    ## HELPER FUNCTION: not agnostic of specific dataset configs
    def emd_by_class(self, noisy_pos, clean_pos, clean_ang, clean_sha, nobjs):
        """ For each scene, for each object, assign it a target object of the same class based on its position.
            Performs earthmover distance assignment based on pos from noisy to target, for instances of one class, 
            and assign ang correspondingly.

            pos/ang/sha: [batch_size, maxnumobj, ang/pos/sha(siz+cla)_dim]
            nobjs: [batch_size]
            clean_sha: noisy and target (clean) should share same shape data (size and class don't change)
        """
        numscene = noisy_pos.shape[0]
        noisy_labels = np.zeros((numscene, self.maxnobj, self.pos_dim+self.ang_dim))
        for scene_i in range(numscene):
            nobj = nobjs[scene_i]
            cla_idx = np.argmax(clean_sha[scene_i, :nobj, self.siz_dim:], axis=1) # (nobj, cla_dim) -> (nobj,) # example: array([ 9, 11, 15,  7,  2, 18, 15])
            for c in np.unique(cla_idx) : # example unique out: array([ 2,  7,  9, 11, 15, 18]) (indices of the 1 in one-hot encoding)
                objs = np.where(cla_idx==c)[0] # 1d array of obj indices whose class is c # example: array([2, 6]) for c=15
                p1 = [tuple(pt) for pt in noisy_pos[scene_i, objs, :]] # array of len(objs) tuples
                p2 = [tuple(pt) for pt in clean_pos[scene_i, objs, :]] # array of len(objs) tuples
                chair_assignment, chair_assign_ang = earthmover_assignment(p1, p2, clean_ang[scene_i, objs, :]) # len(objs)x2: assigned pos for each pt in p1 (in that order)
                noisy_labels[scene_i, objs, 0:self.pos_dim] = np.array(chair_assignment)
                noisy_labels[scene_i, objs, self.pos_dim:self.pos_dim+self.ang_dim] = np.array(chair_assign_ang)
        # NOTE: noisy_labels[scene_i, nobj:, :] left as 0; each obj assigned exactly once
        return noisy_labels
    
    def add_gaussian_gaussian_noise_by_class(self, classname, noisy_orig_pos, noisy_orig_sha, noise_level_stddev=0.1):
        noisy_pos = np.copy(noisy_orig_pos) # [batch_size, maxnobj, pos]
        
        for scene_i in range(noisy_orig_pos.shape[0]):
            nobj, cla_idx = TDSGFront.parse_cla(noisy_orig_sha[scene_i, :, self.siz_dim:]) # generated input, never perturbed
            for o_i in range(nobj):
                if classname in self.object_types[cla_idx[o_i]]: # only add noise for chairs
                    noisy_pos[scene_i:scene_i+1, o_i:o_i+1, :] = np_add_gaussian_gaussian_noise(noisy_orig_pos[scene_i:scene_i+1, o_i:o_i+1, :], noise_level_stddev=noise_level_stddev)

        return noisy_pos
    
    def clever_add_noise(self, noisy_orig_pos, noisy_orig_ang, noisy_orig_sha, noisy_orig_nobj, noisy_orig_fpoc, noisy_orig_nfpc, noisy_orig_vol, 
                         noise_level_stddev, angle_noise_level_stddev, weigh_by_class=False, within_floorplan=False, no_penetration=False, max_try=None, pen_siz_scale=0.92):
        """ noisy_orig_pos/ang/sha: [batch_size, maxnobj, pos_dim/ang_dim/sha_dim]
            noisy_orig_fpoc:        [batch_size, maxnfpoc, pos_dim]
            noisy_orig_vol:         used only if weigh_by_class
        """
        if not weigh_by_class and not within_floorplan and not no_penetration:
            # NOTE: each scene has zero-mean gaussian distributions for noise
            # noisy_pos = add_gaussian_gaussian_noise_by_class("chair', noisy_orig_pos, noisy_orig_sha, noise_level_stddev=noise_level_stddev)
            noisy_pos = np_add_gaussian_gaussian_noise(noisy_orig_pos, noise_level_stddev=noise_level_stddev)
            noisy_ang = np_add_gaussian_gaussian_angle_noise(noisy_orig_ang, noise_level_stddev=angle_noise_level_stddev) # [batch_size, maxnumobj, 2], stay normalized
            noisy_pos, noisy_ang = TDSGFront.reset_padding(noisy_orig_nobj, noisy_pos), TDSGFront.reset_padding(noisy_orig_nobj, noisy_ang) # [batch_size, maxnumobj, dim]
            return noisy_pos, noisy_ang
                
        if max_try==None: 
            max_try=1 # weigh_by_class only, exactly 1 iteration through while loop (always reach break in first iter)
            if within_floorplan: max_try+= 1000
            if no_penetration: max_try+= 2000 # very few in range 300-1500 for bedroom

        noisy_pos = np.copy(noisy_orig_pos) # the most up to date arrangement, with 0 padded
        noisy_ang = np.copy(noisy_orig_ang)
        obj_noise_factor = 1 # overriden if weighing by class
        
        if weigh_by_class: obj_noise_factors = 1/np.sqrt((noisy_orig_vol+0.00001)*2) # [batch_size, maxnobj, 1], intuition: <1 for large objects, >1 for small objects
            # Purpose of *2: so not as extreme: vol<2, factor < 1/vol (not too large); > 2, factor > 1/vol (not too small)
        # Ending value are inf, but not used as we only consider up until nobj
        
        for scene_i in range(noisy_orig_pos.shape[0]): # each scene has its own noise level
            # NOTE: each scene has zero-mean gaussian distributions for noise and noise level
            scene_noise_level = abs(np.random.normal(loc=0.0, scale=noise_level_stddev)) # 68% in one stddev
            scene_angle_noise_level = abs(np.random.normal(loc=0.0, scale=angle_noise_level_stddev))

            parse_nobj, cla_idx = TDSGFront.parse_cla(noisy_orig_sha[scene_i, :, self.siz_dim:]) # generated input, never perturbed
            obj_indices = list(range(noisy_orig_nobj[scene_i]))
            random.shuffle(obj_indices) # shuffle in place
            for obj_i in obj_indices: # 0 padding unchanged
                # if "chair" not in self.object_types[cla_idx[obj_i]]: continue # only add noise for chairs

                if weigh_by_class: obj_noise_factor = obj_noise_factors[scene_i, obj_i, 0] # larger objects have smaller noise
                # print(f"\n--- obj_i={obj_i}: nf={nf}")
                try_count = -1
                while True:
                    try_count += 1
                    if try_count >= max_try: 
                        # print(f"while loop counter={try_count}")
                        break
                    obj_noise = obj_noise_factor * np.random.normal(size=(noisy_orig_pos[scene_i, obj_i:obj_i+1, :]).shape, loc=0.0, scale=scene_noise_level) # [1, pos_dim]
                    new_o_pos = noisy_orig_pos[scene_i, obj_i:obj_i+1, :] + obj_noise # [1, pos_dim]

                    obj_angle_noise = obj_noise_factor * np.random.normal(size=(1,1), loc=0.0, scale=scene_angle_noise_level) 
                    new_o_ang = np_rotate(noisy_orig_ang[scene_i, obj_i:obj_i+1, :], obj_angle_noise) # [1, ang_dim=2]

                    if within_floorplan or no_penetration: # if can skip, will always break
                        if not self.is_valid( obj_i, np.copy(new_o_pos), np.copy(new_o_ang),
                                            np.copy(noisy_pos[scene_i, :noisy_orig_nobj[scene_i], :]), np.copy(noisy_ang[scene_i, :noisy_orig_nobj[scene_i], :]),  # latest state
                                            np.copy(noisy_orig_sha[scene_i, :noisy_orig_nobj[scene_i], :]), np.copy(noisy_orig_fpoc[scene_i, :noisy_orig_nfpc[scene_i], :]), 
                                            within_floorplan=within_floorplan, no_penetration=no_penetration, pen_siz_scale=pen_siz_scale):
                            continue # try regenerating noise

                    # reached here means passed checks
                    noisy_pos[scene_i, obj_i:obj_i+1, :] = new_o_pos
                    noisy_ang[scene_i, obj_i:obj_i+1, :] = new_o_ang

                    # print(f"break @ try_count={try_count}")
                    break # continue to next object
        return noisy_pos, noisy_ang
    
    def is_valid(self, o_i, o_pos, o_ang, scene_pos, scene_ang, scene_sha, scene_fpoc, within_floorplan=True, no_penetration=True, pen_siz_scale=0.92):
        """ A object's pos + ang is valid if the object's bounding box does not intersect with any floor plan wall or other object's bounding box edge.
            Note this function modifies the input arguments in place.

            o_i: scalar, the index of the object of interest in the scene
            o_{pos, ang}: [1, dim], information about the object being repositioned
            scene_{pos, ang, sha}: [nobj, dim], info about the rest of the scene, without padding, the obj_ith entry of pos and ang is skipped
            scene_fpoc: [nfpoc, pos_dim], without padding, ordered (consecutive points form lines).
            pen_siz_scale: to allow for some minor intersection (respecting ground truth dataset)
        """
        # Dnormalize data to the same scale: [-3, 3] (meters) for both x and y(z) axes.
        room_size = np.array(self.room_size) #[x, y, z]
        o_pos = o_pos*room_size[[0,2]]/2  #[1, dim]
        scene_fpoc = scene_fpoc* room_size[[0,2]]/2 # from [-1,1] to [-3,3]
        scene_pos = scene_pos*room_size[[0,2]]/2 # [nobj, pos_dim]
        scene_ang = trig2ang(scene_ang) #[nobj, 1], in [-pi, pi]
        scene_siz = (scene_sha[:,:self.siz_dim] +1) * (room_size[[0,2]]/2)  # [nobj, siz_dim], bbox len 
    
        # check intersection with floor plan
        if within_floorplan:
            # check if obj o's pos and corners is outside floor plan's convex bounds
            fp_x_min, fp_x_max, fp_y_min, fp_y_max = TDSGFront.get_xyminmax(scene_fpoc)
            if ((o_pos[0,0]<fp_x_min) or (o_pos[0,0]>fp_x_max)or (o_pos[0,1]<fp_y_min) or (o_pos[0,1]>fp_y_max)): return False
            o_corners, o_bboxedge = TDSGFront.get_objbbox_corneredge(o_pos[0], trig2ang(o_ang), scene_siz[o_i]) # cor=(4,2)
            o_cor_x_min, o_cor_x_max, o_cor_y_min, o_cor_y_max = TDSGFront.get_xyminmax(o_corners)
            if ((o_cor_x_min<fp_x_min) or (o_cor_x_max>fp_x_max)or(o_cor_y_min)<fp_y_min) or (o_cor_y_max>fp_y_max): return False

            # check for intersection of boundaries
            for wall_i in range (len(scene_fpoc)):
                fp_pt1, fp_pt2 = scene_fpoc[wall_i], scene_fpoc[(wall_i+1)%len(scene_fpoc)]

                # Entirely out of bounds for each line (especially for concave shapes): scene_fpoc ordered counterclockwisely starting from bottom left corner 
                if(fp_pt1[0] == fp_pt2[0]): # vertical
                    if (fp_pt1[1]>=fp_pt2[1]): # top to bottom, right edge
                        if o_cor_x_min >= fp_pt1[0]: return False
                    else: # bottom to top, left edge
                        if o_cor_x_max <= fp_pt1[0]: return False
                if(fp_pt1[1] == fp_pt2[1]): # horizontal
                    if (fp_pt1[0]>=fp_pt2[0]): # from right to left, bottom edge
                        if o_cor_y_max <= fp_pt1[1]: return False
                    else: # from left to right top edge
                        if o_cor_y_min >= fp_pt1[1]: return False

                for edge_i in range(4): # obj is rectangular bounding box
                    if do_intersect(o_bboxedge[edge_i][0], o_bboxedge[edge_i][1], fp_pt1, fp_pt2):
                        return False

        # check intersection with each of the other objects
        if no_penetration:
            o_scale_corners, o_scale_bboxedge = TDSGFront.get_objbbox_corneredge(o_pos[0], trig2ang(o_ang), scene_siz[o_i]*pen_siz_scale)
            o_scale_cor_x_min, o_scale_cor_x_max, o_scale_cor_y_min, o_scale_cor_y_max = TDSGFront.get_xyminmax(o_scale_corners)

            for other_o_i in range (scene_pos.shape[0]):
                if other_o_i == o_i: continue # do not compare against itself
                other_scale_cor, other_scale_edg = TDSGFront.get_objbbox_corneredge(scene_pos[other_o_i], scene_ang[other_o_i:other_o_i+1,:], scene_siz[other_o_i]*pen_siz_scale)
                other_scale_cor_x_min, other_scale_cor_x_max, other_scale_cor_y_min, other_scale_cor_y_max = TDSGFront.get_xyminmax(other_scale_cor)
                
                # check entire outside one another
                if ((o_scale_cor_x_max<=other_scale_cor_x_min) or (o_scale_cor_x_min>=other_scale_cor_x_max) or
                    (o_scale_cor_y_max<=other_scale_cor_y_min) or (o_scale_cor_y_min>=other_scale_cor_y_max)):
                   continue # go check next obj

                # check if one is inside the other:
                if ((other_scale_cor_x_min <= o_scale_cor_x_min <= other_scale_cor_x_max) and (other_scale_cor_x_min <= o_scale_cor_x_max <= other_scale_cor_x_max) and
                    (other_scale_cor_y_min <= o_scale_cor_y_min <= other_scale_cor_y_max) and (other_scale_cor_y_min <= o_scale_cor_y_max <= other_scale_cor_y_max)):
                    return False
                if ((o_scale_cor_x_min <= other_scale_cor_x_min <= o_scale_cor_x_max) and (o_scale_cor_x_min <= other_scale_cor_x_max <= o_scale_cor_x_max) and
                    (o_scale_cor_y_min <= other_scale_cor_y_min <= o_scale_cor_y_max) and (o_scale_cor_y_min <= other_scale_cor_y_max <= o_scale_cor_y_max)):
                    return False
                # check if edges intersect
                for edge_i in range(4):
                    for other_edge_i in range(4):
                        if do_intersect(o_scale_bboxedge[edge_i][0], o_scale_bboxedge[edge_i][1], 
                                        other_scale_edg[other_edge_i][0], other_scale_edg[other_edge_i][1]):
                            return False

        return True
    
    def gen_random_selection(self, batch_size, data_partition='trainval'):
        if data_partition=='trainval':
            total_data_count = self.data_tv['pos'].shape[0]
        elif data_partition=='test':
            total_data_count = self.data_test['pos'].shape[0]
        elif data_partition=='all':
            total_data_count = self.data_tv['pos'].shape[0] + self.data_test['pos'].shape[0]
        return np.random.choice(total_data_count, size=batch_size, replace=False)   # False: each data can be selected once # (batch_size,)
    
    def gen_stratified_selection(self, n_to_select, data_partition='test'):
        """ Only makes sense for augmented dataset. Select at least 1 from each original scene, and the (n_to_select-n_original_scene) scenes
            are selected randomly from the remaining unselected scenes.
        """
        if data_partition=='trainval':
            data = self.data_tv
        elif data_partition=='test':
            data = self.data_test
        elif data_partition=='all':
            pass # TODO

        # first 'stratify the dataset'
        originalscene2id = {} # {scenedir: [10,2381,103,800]}
        for i , scenedir in enumerate(data['scenedirs']): # 00ecd5d3-d369-459f-8300-38fc159823dc_SecondBedroom-6249_0
            if scenedir[:-2] in originalscene2id:
                originalscene2id[scenedir[:-2]].append(i)
            else:
                originalscene2id[scenedir[:-2]]= [i] # originalscene2id[data['scenedirs'][24][:-2]]) = [24, 25, 26, 27]
        
        grouped_ids = []
        for originalscene in originalscene2id:
            grouped_ids.append(originalscene2id[originalscene])
        grouped_ids = np.array(grouped_ids)  # (224, 4) for bedroom, if not augmented, then (224, 1)
        
        selection=np.random.randint(grouped_ids.shape[1], size=[grouped_ids.shape[0]]) # for each row, pick 1 element
        guaranteed_selection = grouped_ids[range(grouped_ids.shape[0]), selection] # [grouped_ids.shape[0], ]
        if n_to_select < guaranteed_selection.shape[0]:
            return np.random.choice(guaranteed_selection, size=n_to_select, replace=False)  # False: each data can be selected once 

        remaining_unselected = np.delete(np.arange(data['scenedirs'].shape[0]), guaranteed_selection) # total nscene - guaranteed_selection.shape[0]
        remaining_selection = np.random.choice(remaining_unselected, size=n_to_select-guaranteed_selection.shape[0], replace=False)
        return np.append(guaranteed_selection, remaining_selection) # (n_to_select, )
    
    def _gen_3dfront_batch_preload(self, batch_size, data_partition='trainval', use_floorplan=True, random_idx=None):
        """ Reads from preprocessed data npz files (already normalized) to return data for batch_size number of scenes.
            Variable data length is dealt with through padding with 0s at the end.

            random_idx: if given, selects these designated scenes from the set of all trainval or test data.
            
            Returns:
            batch_scenepaths: [batch_size], contains full path to the directory named as the scenepath (example
                             scenepath = '<scenedir>/004f900c-468a-4f70-83cc-aa2c98875264_SecondBedroom-27399')
            batch_nbj: [batch_size], contains numbers of objects for each scene/room.

            batch_pos: position has size [batch_size, maxnumobj, pos_dim]=[x, y], where [:,0:2,:] are the 2 tables,
                       and the rest are the chairs.
            batch_ang: [batch_size, maxnumobj, ang_dim=[cos(th), sin(th)] ]
            batch_sha: [batch_size, maxnumobj, siz_dim+cla_dim], represents bounding box lengths and
                       class of object/furniture with one hot encoding.
            
            batch_vol: [batch_size, maxnumobj], volume of each object's bounding box (in global absolute scale).

            floor plan representations:
               batch_fpoc:   [batch_size, maxnfpoc, pos_dim]. Floor plan ordered corners. For each scene, have a list of
                             ordered (clockwise starting at bottom left [-x, -y]) corner points of floor plan contour,
                             where consecutive points form a line. Padded with 0 at the end. Normalized to [-1, 1] in 
                             write_all_data_summary_npz.
               batch_nfpc:   [batch_size], number of floor plan corners for each scene
               batch_fpmask: [batch_size, 256, 256, 3]
               batch_fpbpn:  [batch_size, self.nfpbp=250, 4]. floor plan boundary points and normals, including corners.
                             [:,:,0:2] normalized in write_all_data_summary_npz                   
        """
        random_idx = self.gen_random_selection(batch_size, data_partition) if random_idx is None else random_idx
        data = self.data_tv 
        if data_partition=='test': data = self.data_test
        if data_partition=='all':  
            data = dict(self.data_tv)
            data_test = dict(self.data_test)
            for key in data: data[key] = np.concatenate([data[key], data_test[key]], axis=0)

        batch_scenepaths = []  
        for data_i in range(batch_size): 
            s = data['scenedirs'][random_idx[data_i]] # a6704fd9-02c2-42a6-875c-723b26a8048a_MasterBedroom-45545
            batch_scenepaths.append(os.path.join(self.scene_dir, s))
        batch_scenepaths = np.array(batch_scenepaths) # [] # numpy array of strings

        batch_nbj = data['nbj'][random_idx] # [] -> (batch_size,)

        batch_pos = data['pos'][random_idx] # np.zeros((batch_size, self.maxnobj, self.pos_dim)) # batch_size, maxnobj, pos_dim
        batch_ang = data['ang'][random_idx] # np.zeros((batch_size, self.maxnobj, self.ang_dim))
        batch_siz = data['siz'][random_idx] # np.zeros((batch_size, self.maxnobj, self.siz_dim)) # shape
        batch_cla = data['cla'][random_idx] # np.zeros((batch_size, self.maxnobj, self.cla_dim)) # shape
        batch_sg  = data["sg"][random_idx] # np.zeros((batch_size, self.maxnobj,)) # shape
        batch_vol = data['vol'][random_idx] # np.zeros((batch_size, self.maxnobj, 1)) # shape

        batch_fpoc, batch_nfpc, batch_fpmask, batch_fpbpn = None, [], None, None
        if use_floorplan:
            # floor plan representation 1: floor plan ordered corners
            batch_fpoc = data['fpoc'][random_idx] # np.zeros((batch_size, self.maxnfpoc, self.pos_dim))
            batch_nfpc = data['nfpc'][random_idx] # [] (batch_size,)
            
            # floor plan representation 2: binary mask (1st channel is remapped_room_layout=drawing ctr/rescaled fpoc on empty mask)
            batch_fpmask = np.zeros((batch_size, 256, 256, 3))  # 1 3-channel mask per scene
            xy = generate_pixel_centers(256,256) / 128 -1 # [0 (0.5), 256 (255.5)] -> [0,2] -> [-1,1] # in the same coord system as vertices 
            for data_i in range(batch_size): 
                random_i = random_idx[data_i]
                new_contour_mask = np.zeros((256,256,1))
                ctr = data['ctr'][random_i] # (51, 1, 2)
                ctr = np.expand_dims(ctr[np.any(ctr, axis=2)], axis=1) # (numpt, none->1, 2), kept non-zero rows
                cv.drawContours(new_contour_mask, [ctr.astype(np.int32)], -1 , (255,255,255), thickness=-1) # thickness < 0 : fill
                batch_fpmask[data_i] = np.concatenate([new_contour_mask, np.copy(xy)], axis=2) #(256,256,1+2=3)

            # floor plan representation 3: floor plan boundary points & their normals
            batch_fpbpn = data['fpbpn'][random_idx] # (batch_size, self.nfpbp=250, 4)

        batch_sha = np.concatenate([batch_siz, batch_cla], axis=2)  # [batch_size, maxnumobj, 2+22=24]
        return batch_scenepaths, batch_nbj, batch_pos, batch_ang, batch_sha, batch_vol, batch_fpoc, batch_nfpc, batch_fpmask, batch_fpbpn, batch_sg
        

    

    


    
