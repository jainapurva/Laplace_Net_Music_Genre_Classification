import torchvision.transforms as transforms
from .augmentations import RandAugment
from .utils import export
import os

@export
def gtzan():
    channel_stats = dict(mean=[0.8914, 0.8822, 0.8465],
                         std=[0.6470,  0.6435,  0.6616])
    
    #channel_stats = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470,  0.2435,  0.2616])
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])


    myhost = os.uname()[1]
    data_dir = 'data-local/images/gtzan/by-image'

    print("Using GTZAN from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }
