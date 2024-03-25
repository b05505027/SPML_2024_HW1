from dataset import CIFAR100Eval
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import freeze_model, compare_images, tensor_to_image
from tqdm import tqdm

''' Step 0: load the dataset and the model '''
dataset = CIFAR100Eval(root='./cifar-100_eval', transform=None)
net = ptcv_get_model("resnet20_cifar100", pretrained=True)
freeze_model(net)
net.eval()
criterion = torch.nn.CrossEntropyLoss()



for data in tqdm(dataset):
    ''' Step1: Loading the image and converting it to a PyTorch tensor Variable '''
    img, label, PATH = data # load PIL images
    img = np.array(img) # convert to numpy array
    img = torch.from_numpy(img) # convert to PyTorch tensor
    img = img.float() # convert to float tensor

    ''' Step2: Applying transformations to the image '''
    img.requires_grad = True # set requires_grad to True for adversarial attack
    img_p = img.permute(2, 0, 1) # convert from HWC to CHW
    img_zo = img_p / 255.0 # normalize to [0, 1]
    img_norm = transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))(img_zo)

    ''' Step3: Make prediction from the image'''
    # Use vanilla gradient descent for optimization
    optimizer = torch.optim.SGD([img], lr=0.01)
    prediction = net(img_norm.unsqueeze(0))

    ''' Step4: Calculate the loss and backpropagate '''
    loss = -criterion(prediction, torch.tensor(label).unsqueeze(0))
    loss.backward()
    epsilon = 0.01
    img.grad = img.grad.sign()
    img = img + 8 * img.grad

    ''' Step5: Inverting the transformations and saving the image '''
    # from CHW to HWC
    img = img.detach().cpu()
    img = img.clamp(0, 255)
    img = img.type(torch.uint8)
    NEW_PATH = PATH.replace('cifar-100_eval', 'cifar-100_FGSM')
    tensor_to_image(img, NEW_PATH)
    if compare_images(PATH, NEW_PATH)[0] != True:
        # throw error if the image is not within the threshold
        raise ValueError("The image crosses the threshold") 
