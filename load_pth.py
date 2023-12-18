import torch

superglue = torch.load("/home/zhenyu/Desktop/OpenSource/SuperGlue-pytorch/checkpoint/10-22-17-23-04/model_epoch_best.pth")

superglue = superglue.eval()
superglue = superglue.cpu()
ckpt_model_filename = "/home/zhenyu/Desktop/OpenSource/SuperGlue-pytorch/checkpoint/10-22-17-23-04/sg_fk-best.pth"
torch.save(superglue.state_dict(), ckpt_model_filename)
