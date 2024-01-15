from data.TDSGFront import TDSGFront
from model.transformer import TransformerWrapper

dataset = TDSGFront()
input = dataset.gen_3dfront(batch_size=16)

# model = TransformerWrapper()
# print(model)