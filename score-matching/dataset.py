# NOTE: To use FFHQ dataset, you need to download the dataset and place images in the folder "data/ffhq/ffhq/thumbnails128x128/" in the root directory.
# The "thumbanils128x128" folder should contain 70,000 images of size 128x128, and no other files of any type.

import torch
import torchvision

def get_data(dataset_name, batch_size):
    
    if dataset_name == "cifar10":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        class_names = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        dataset = torchvision.datasets.CIFAR10("../data/cifar10", train=True, download=True, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    elif dataset_name == "ffhq_96":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(96),
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder("../data/ffhq/thumbnails128x128/", transform=transform)
        data_loader = torch.utils.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        
    elif dataset_name == "ffhq_128":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder("../data/ffhq/thumbnails128x128/", transform=transform)
        data_loader = torch.utils.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
 
    else:
        raise NotImplementedError()

    return data_loader


def get_dimensions(dataset_name):
    
    if dataset_name == "cifar10":
        image_size = 32
        channels = 3   
        dimensions = [32, 64, 128, 256]
        embedding_size = 256
        
    elif dataset_name == "ffhq_96":
        image_size = 96
        channels = 3
        dimensions = [96, 192, 384, 768]
        embedding_size = 768

    elif dataset_name == "ffhq_128":
        image_size = 128
        channels = 3
        dimensions = [128, 256, 512, 1024]
        embedding_size = 1024
        
    else:
        raise ValueError("Dataset not supported yet")
    return image_size, channels, dimensions, embedding_size