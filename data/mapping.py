bedroom_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe", "start", "end"]
livingroom_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "start", "end"]
library_idx = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair", "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "dressing_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", "round_end_table", "shelf", "stool", "wardrobe", "wine_cabinet", "start", "end"]

sup_cat2idx = {
    "seating" : 0,
    "storage" : 1,
    "surface" : 2,
    "lighting" : 3,
    "decor" : 4
}

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