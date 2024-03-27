from dataset import CIFAR100Eval
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import freeze_model, compare_images, tensor_to_image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLossFunction(nn.Module):
    def __init__(self, c=0.5):
        """
        Initialize the custom loss function.
        :param c: The weight of the second term in the loss function.
        """
        super(CustomLossFunction, self).__init__()
        self.c = c

    def forward(self, w, x, logits, targets):
        """
        Forward pass of the custom loss function.
        :param w: The node variable.
        :param x: The input images of size (batch, C, H, W).
        :param logits: The logits from the neural network of size (batch, class_numbers).
        :param targets: The ground truth labels of size (batch, 1).
        """
        # Ensure targets are in the correct shape for gathering
        targets = targets.squeeze(-1)
        
        # Transform w back to [0, 255] range
        w_transformed = 255 / 2 * (torch.tanh(w) + 1)

        # Compute the L2 loss between transformed w and original x
        loss1 = torch.mean((w_transformed - x) ** 2)
        
        # Compute the margin loss using logits and targets
        true_class_scores = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        max_others = logits.clone()
        max_others[range(max_others.shape[0]), targets] = -float('inf')
        max_other_scores = torch.max(max_others, dim=1)[0]


        margin_loss = torch.maximum(true_class_scores - max_other_scores, torch.tensor(-10))

        loss2 = self.c * torch.mean(margin_loss)

        total_loss = loss1 + loss2


        return total_loss




''' Step 0: load the dataset and the model '''
dataset = CIFAR100Eval(root='./cifar-100_eval', transform=None)


nets = [
    ptcv_get_model("resnet56_cifar100", pretrained=True),
    ptcv_get_model("resnet20_cifar100", pretrained=True),
    ptcv_get_model("wrn16_10_cifar100", pretrained=True),
    ptcv_get_model("seresnet164bn_cifar100", pretrained=True),
    ptcv_get_model("resnext29_32x4d_cifar100", pretrained=True),
    ptcv_get_model("resnet110_cifar100", pretrained=True),
    ptcv_get_model("pyramidnet110_a270_cifar100", pretrained=True),
    ptcv_get_model("densenet100_k24_cifar100", pretrained=True),

]

for net in nets:
    freeze_model(net)
    net.eval()

criterion = CustomLossFunction(c=8)



for data in tqdm(dataset):

    ''' Step1: Loading the image '''
    img, label, PATH = data # load PIL images
    img = np.array(img) # convert to numpy array
    img = torch.from_numpy(img) # convert to PyTorch tensor
    img = img.float() # convert to float tensor
    origin_tensor = img.clone()

    ''' Step2: Set the valid range of the variable w '''
    # w_lower = torch.atanh(2 * torch.max((origin_tensor - 8), 0) / 255 - 1)
    # w_upper = torch.atanh(2 * torch.min((origin_tensor + 8), 255) / 255 - 1)


    w_lower = torch.atanh(2 * torch.maximum(origin_tensor - 7.9, torch.tensor(0)) / 255 - 1)
    w_upper = torch.atanh(2 * torch.minimum(origin_tensor + 7.9, torch.tensor(255)) / 255 - 1)


    ''' Step3: make the image a variable and load the optimizer '''
    w = torch.atanh(2 * img / 255 - 1)
    w.requires_grad = True
    optimizer = torch.optim.AdamW([w], lr=5e-3)  

    for i in tqdm(range(51), total=50):
        
        ''' Step4: Applying transformations to the image and calculate w'''
        img = 255 / 2 * (torch.tanh(w) + 1)
        img_p = img.permute(2, 0, 1) # convert from HWC to CHW
        img_zo = img_p / 255.0 # normalize to [0, 1]
        img_norm = transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))(img_zo)

        ''' Step5: Make prediction from the image'''
        # prediction = net(img_norm.unsqueeze(0))
        # prediction2 = net2(img_norm.unsqueeze(0))


        # randomly select 3 models to make prediction
        selection = np.random.choice(len(nets), 3, replace=False)
        predictions = []
        for k in selection:
            predictions.append(nets[k](img_norm.unsqueeze(0)))
        prediction = torch.mean(torch.stack(predictions), dim=0)

        
        ''' Step6: Calculate the loss and backpropagate '''
        loss = criterion(w, origin_tensor, prediction, torch.tensor(label).unsqueeze(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        


        ''' Step8: Inverting the transformations and saving the image '''
        if i % 25 == 0:
            # from CHW to HWC
            w_clone = w.clone().detach().cpu()

            #print('w before clipping', w_clone)
            w_clone = w_clone.clamp(min=w_lower, max=w_upper)
            #print('w after clipping', w_clone)
            img_clone = 255 / 2 * (torch.tanh(w_clone) + 1)
            img_clone = img_clone.clamp(0, 255)
            img_clone = img_clone.type(torch.uint8)
            NEW_PATH = PATH.replace('cifar-100_eval', 'cifar-100_CW_mix')
            # NEW_PATH = 'imgage_{}.png'.format(i)
            tensor_to_image(img_clone, NEW_PATH)
            
            flag, val = compare_images(PATH, NEW_PATH)
            if flag != True:
                # throw error if the image is not within the threshold
                #print(torch.tanh(w_clone).max())
                raise ValueError("The image crosses the threshold") 

            else:
                print('Image is within the threshold', val)
            
            # input()

