from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from utils import get_dataloader, freeze_model
from metrics import Top1Error
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='Evaluate the models')
parser.add_argument('-m', '--mode', type=str, default='eval', help='The mode to evaluate the model')
args = parser.parse_args()

''' Step 0: load the dataset and the models '''
dataloader = get_dataloader(mode=args.mode, batch_size=32)
net_names = [
    "resnet56_cifar100",
    "resnet20_cifar100",
    "wrn16_10_cifar100",
    "seresnet164bn_cifar100",
    "resnext29_32x4d_cifar100",
    "resnet110_cifar100",
    "pyramidnet110_a270_cifar100",
    "densenet100_k24_cifar100",
]
nets = [ptcv_get_model(name, pretrained=True) for name in net_names]
for net in nets:
    freeze_model(net)
    net.eval()

for name_index, net in enumerate(nets):
    print(f"Evaluating model: {net_names[name_index]}")

    ''' Step 1: Set the loss function and the evaluation metric '''
    metric = Top1Error()
    criterion = torch.nn.CrossEntropyLoss()
    acc_loss = []

    ''' Step 2: Evaluate the model '''
    for item in tqdm(dataloader):
        data, label, path = item
        prediction = net(data)
        metric.update(label, prediction)
        loss = criterion(prediction, label)
        acc_loss.append(loss.item())

    ''' Step 3: Get the evaluation metric '''
    name, value = metric.get()
    avg_loss = sum(acc_loss) / len(acc_loss)
    print(f"Evalutaion Metric: {name} {value}")
    print(f"Average Loss: {avg_loss}")

    

