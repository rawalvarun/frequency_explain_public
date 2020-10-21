import torchvision.models as models

map_config_to_model = {}

map_config_to_model['resnet18'] = models.resnet18(pretrained=True)
map_config_to_model['alexnet'] = models.alexnet(pretrained=True)
map_config_to_model['vgg16'] = models.vgg16(pretrained=True)
map_config_to_model['squeezenet'] = models.squeezenet1_0(pretrained=True)
map_config_to_model['densenet'] = models.densenet161(pretrained=True)
map_config_to_model['inception'] = models.inception_v3(pretrained=True)
map_config_to_model['googlenet'] = models.googlenet(pretrained=True)
map_config_to_model['shufflenet'] = models.shufflenet_v2_x1_0(pretrained=True)
map_config_to_model['mobilenet'] = models.mobilenet_v2(pretrained=True)
map_config_to_model['resnext50_32x4d'] = models.resnext50_32x4d(pretrained=True)
map_config_to_model['wide_resnet50_2'] = models.wide_resnet50_2(pretrained=True)
map_config_to_model['mnasnet'] = models.mnasnet1_0(pretrained=True)
