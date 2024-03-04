from torch.utils.data import DataLoader
from dataset import CIFAR100MetaInfo

def get_dataloader():
    ds_meta = CIFAR100MetaInfo()
    transform = ds_meta.val_transform(ds_meta)
    dataset = ds_meta.dataset_class(root=ds_meta.root_dir_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    return dataloader
