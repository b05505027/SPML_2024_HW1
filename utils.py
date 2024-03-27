from torch.utils.data import DataLoader
from dataset import CIFAR100MetaInfo, CIFAR100FGSMMetaInfo, CIFAR100CWMetaInfo
from PIL import Image
import numpy as np


def get_dataloader(mode, batch_size, size=None):
    # check if the mode is in ['fgsm', 'pgd', 'eval']
    if mode not in ['fgsm', 'cw', 'eval']:
        raise ValueError(f"Invalid mode: {mode}")
    if mode == 'eval':
        ds_meta = CIFAR100MetaInfo()
    elif mode == 'cw':
        ds_meta = CIFAR100CWMetaInfo()
    elif mode == 'fgsm':
        ds_meta = CIFAR100FGSMMetaInfo()

    transform = ds_meta.val_transform(ds_meta)
    dataset = ds_meta.dataset_class(root=ds_meta.root_dir_path, transform=transform, size=size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return dataloader

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def compare_images(image_path1, image_path2):
    """
    Compare two images to check if the difference between each corresponding pixel is within 8.
    
    Args:
    - image_path1: Path to the first image file.
    - image_path2: Path to the second image file.
    
    Returns:
    - A boolean indicating whether all pixel differences are within 8.
    """
    # Load the images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    
    
    # Convert images to NumPy arrays
    img1_np = np.array(img1).astype(np.int32)
    img2_np = np.array(img2).astype(np.int32)

    
    # Check if the images have the same shape
    if img1_np.shape != img2_np.shape:
        raise ValueError("Images do not have the same size or number of channels")

    # Compute the absolute difference between the images
    diff = (img1_np - img2_np)
    # Check if all differences are within 8
    within_threshold = np.all(diff <= 8)

    # If not all are within the threshold, find the maximum difference
    max_diff = np.max(diff)
    # get the index of the maximum difference
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print('max difd at index:', max_diff_idx)
    # value of img1 and img2 at the max_diff_idx
    print('img1 value:', img1_np[max_diff_idx])
    print('img2 value:', img2_np[max_diff_idx])
    
    return (within_threshold, max_diff)

def tensor_to_image(tensor, save_path):
    """
    Convert a PyTorch tensor with shape (H, W, C) and values in [0, 255] to an image and save it.
    
    Args:
    - tensor: A PyTorch tensor to be converted.
    - save_path: Path where the image will be saved.
    """
    # Ensure the tensor is in CPU memory and detach it from the computation graph
    tensor = tensor.detach().cpu()
    
    # Convert the tensor to a PIL image
    # The tensor should be uint8 and in [0, 255]. If not, you must convert it accordingly.
    # Assuming the tensor is already in the correct format:
    image = Image.fromarray(tensor.numpy().astype('uint8'))
    
    # Save the image
    image.save(save_path)

