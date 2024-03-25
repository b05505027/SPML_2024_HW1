from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from utils import get_dataloader, freeze_model
from metrics import Top1Error
from tqdm import tqdm


''' Step 0: load the dataset and the model '''
dataloader = get_dataloader(mode='fgsm', batch_size=32)
net = ptcv_get_model("resnet20_cifar100", pretrained=True)
freeze_model(net)
net.eval()

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

    

