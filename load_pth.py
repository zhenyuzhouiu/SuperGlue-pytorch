import torch

superglue = torch.load("./model_epoch_1.pth")

superglue = superglue.eval()
superglue = superglue.cpu()
ckpt_model_filename = "./sg_fingerknuckle_v1.pth"
torch.save(superglue.state_dict(), ckpt_model_filename)
