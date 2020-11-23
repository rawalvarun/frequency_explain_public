from torchvision import models
from torch.nn import Linear, Conv2d


def model_adapt(model, model_type, num_classes=10, requires_grad=True):
    if model_type == 'resnet18' or model_type == 'wide_resnet50_2' or model_type == 'resnext50_32x4d':
        model.fc = Linear(512, num_classes)
        model.adapt_layers = model.fc
    else:
        if model_type == 'alexnet' or model_type == 'vgg16':
            model.classifier[6] = Linear(4096, num_classes)
        elif model_type == 'squeezenet':
            model.classifier[1] = Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif model_type == 'densenet':
            model.classifier = Linear(1024, num_classes)
            model.adapt_layers = model.classifier
        elif model_type == 'mobilenet' or model_type == 'mnasnet':
            model.classifier[1] = Linear(1280, num_classes)

        model.adapt_layers = model.classifier

    for layer in model.adapt_layers.parameters():
        layer.requires_grad = requires_grad
