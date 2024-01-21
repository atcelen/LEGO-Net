from data.TDSGFront import TDSGFront
from data.TDFront import TDFDataset
from model.transformer import TransformerWrapper

# dataset = TDSGFront()
# input = dataset.gen_3dfront(batch_size=16)
# print(input[3])

tdf=TDFDataset("livingroom", use_augment=True)

sceneid = "0b527162-1129-4d0f-9601-1fa2c2b5998e_LivingDiningRoom-7812_1"
input, scenepath = tdf.read_one_scene(scenepath=sceneid)
tdf.visualize_tdf_2d(input, f"TDFront_{sceneid}.jpg", f"Original", traj=None, scenepath=scenepath, show_corner=False, show_fpbpn=False)

# model = TransformerWrapper()
# print(model)