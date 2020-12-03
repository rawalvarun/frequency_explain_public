from torch import nn

from advertorch.attacks import (
    LinfPGDAttack,
    CarliniWagnerL2Attack,
    JacobianSaliencyMapAttack,
    LBFGSAttack,
    GradientSignAttack,
    ElasticNetL1Attack,
    SinglePixelAttack,
    DDNL2Attack
)


def map_config_to_attack(attack_name, model, classes):
    if attack_name == 'CarliniWagnerL2Attack':
        return CarliniWagnerL2Attack(model, num_classes=len(classes),
                                     confidence=0, targeted=False, learning_rate=0.01, binary_search_steps=9, max_iterations=10000,
                                     abort_early=True, initial_const=0.001, clip_min=0.0, clip_max=1.0, loss_fn=None)

    if attack_name == "JacobianSaliencyMapAttack":
        return JacobianSaliencyMapAttack(model, num_classes=len(classes),
                                         clip_min=0.0, clip_max=1.0, loss_fn=None, theta=1.0, gamma=1.0, comply_cleverhans=False)

    if attack_name == "LinfPGDAttack":
        return LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.01,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)

    if attack_name == "LBFGSAttack":
        return LBFGSAttack(model, num_classes=len(classes), batch_size=1, binary_search_steps=9,
                           max_iterations=100, initial_const=0.01, clip_min=0, clip_max=1, loss_fn=None, targeted=False)

    if attack_name == "GradientSignAttack":
        return GradientSignAttack(model, loss_fn=None, eps=0.01, clip_min=0.0, clip_max=1.0, targeted=False)

    if attack_name == "ElasticNetL1Attack":
        return ElasticNetL1Attack(model, num_classes=len(classes), confidence=0, targeted=False, learning_rate=0.01, binary_search_steps=9, max_iterations=10000, abort_early=False, initial_const=0.001, clip_min=0.0, clip_max=1.0, beta=0.01, decision_rule='EN', loss_fn=None)

    if attack_name == "SinglePixelAttack":
        return SinglePixelAttack(model, max_pixels=100, clip_min=0.0, loss_fn=None, clip_max=1.0, comply_with_foolbox=False, targeted=False)

    if attack_name == "DDNL2Attack":
        return DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=0.01, quantize=True, levels=256, clip_min=0.0, clip_max=1.0, targeted=False, loss_fn=None)
