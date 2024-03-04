from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable
from utils import get_dataloader


net = ptcv_get_model("resnet20_cifar100", pretrained=True)

dataloader = get_dataloader()
for i, (data, label) in enumerate(dataloader):
    print(net(data))
    break