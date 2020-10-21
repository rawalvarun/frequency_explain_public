import torchvision

map_config_to_dataset = {}

transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])


map_config_to_dataset["MNIST"] = {
    "train" : torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    "test" : "train" : torchvision.datasets.MNIST(root='../data', train=False, transform=transform, download=True)
}

map_config_to_dataset["STL10"] = {
    "train" : torchvision.datasets.STL10(root='../data',split='train', download=True, transform=transform)
    "test" : torchvision.datasets.STL10(root='../data',split='test', download=True, transform=transform)
}

map_config_to_dataset["Fashion-MNIST"] = {
    "train" : torchvision.datasets.FashionMNIST(root='../data', train=True, transform=transform, download=True)
    "test" : torchvision.datasets.FashionMNIST(root='../data', train=False, transform=transform, download=True)
}

map_config_to_dataset["CIFAR-10"] = {
    "train" : torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=transform),
    "test" : torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
}

map_config_to_dataset["SVHN"] = {
    "train" : torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=transform),
    "test" : torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
}

torchvision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)
