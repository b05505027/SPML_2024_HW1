
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CIFAR100Eval(Dataset):
    def __init__(self,
                 root,
                 size=None,
                 mode="val",
                 transform=None):

        self.root = root
        self.mode = mode
        self.size = size
        self.transform = transform

        # imgages/labels
        image_dir_path = os.path.join(self.root, "images")
        self.images = []
        self.labels = []
        count = 0
        for image_name in os.listdir(image_dir_path):
            # images of format <label>_<id>.png
            label = int(image_name.split("_")[0])
            self.images.append(os.path.join(image_dir_path, image_name))
            self.labels.append(label)
            count += 1
            if self.size and count >= self.size:
                break
    
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.images[index]
    
    def __len__(self):
        return len(self.images)

class DatasetMetaInfo(object):
    """
    Base descriptor of dataset.
    """

    def __init__(self):
        self.use_imgrec = False
        self.label = None
        self.root_dir_name = None
        self.root_dir_path = None
        self.dataset_class = None
        self.dataset_class_extra_kwargs = None
        self.num_training_samples = None
        self.in_channels = None
        self.num_classes = None
        self.input_image_size = None
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.train_use_weighted_sampler = False
        self.val_metric_capts = None
        self.val_metric_names = None
        self.val_metric_extra_kwargs = None
        self.test_metric_capts = None
        self.test_metric_names = None
        self.test_metric_extra_kwargs = None
        self.saver_acc_ind = None
        self.ml_type = None
        self.allow_hybridize = True
        self.train_net_extra_kwargs = None
        self.test_net_extra_kwargs = None
        self.load_ignore_extra = False

class CIFAR100MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CIFAR100MetaInfo, self).__init__()
        self.label = "CIFAR100Eval"
        self.root_dir_name = "cifar-100_eval"
        self.root_dir_path = os.path.join('./', 'cifar-100_eval')
        self.dataset_class = CIFAR100Eval
        self.in_channels = 3
        self.num_classes = 100
        self.input_image_size = (32, 32)
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.val_transform = cifar100_val_transform
        self.ml_type = "imgcls"

class CIFAR100FGSMMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CIFAR100FGSMMetaInfo, self).__init__()
        self.label = "CIFAR100FGSM"
        self.root_dir_name = "cifar-100_FGSM"
        self.root_dir_path = os.path.join('./', 'cifar-100_FGSM')
        self.dataset_class = CIFAR100Eval
        self.in_channels = 3
        self.num_classes = 100
        self.input_image_size = (32, 32)
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.val_transform = cifar100_val_transform
        self.ml_type = "imgcls"

def cifar100_val_transform(ds_metainfo,
                          mean_rgb=(0.4914, 0.4822, 0.4465),
                          std_rgb=(0.2023, 0.1994, 0.2010)):
    assert (ds_metainfo is not None)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])
