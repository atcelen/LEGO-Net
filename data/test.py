from TDSGFront import TDSGFront

dataset = TDSGFront()
input = dataset._gen_3dfront_batch_preload(batch_size=16)
print(input[-1].shape)