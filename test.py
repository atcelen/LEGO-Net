from data.TDSGFront import TDSGFront
from data.TDFront import TDFDataset
from model.transformer import TransformerWrapper

# dataset = TDSGFront()
# input = dataset.gen_3dfront(batch_size=16)
# print(input[3])

tdf=TDFDataset(room_type="livingroom", use_augment=True)
# tdf = TDSGFront()

sceneid = "ce3aa96b-a6f4-455d-8a83-2a61253c6fec_LivingDiningRoom-2432_2"
input, scenepath = tdf.read_one_scene(scenepath=sceneid)
tdf.visualize_tdf_2d(input, f"TDFront_{sceneid}.jpg", f"Original", traj=None, scenepath=scenepath, show_corner=False, show_fpbpn=False)

# model = TransformerWrapper()
# print(model)