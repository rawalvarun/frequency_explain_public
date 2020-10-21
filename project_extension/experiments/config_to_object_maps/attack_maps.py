
from advertorch.attacks import (
    LinfPGDAttack, 
    CarliniWagnerL2Attack, 
    JacobianSaliencyMapAttack, 
    LBFGSAttack,
    
from advertorch.attacks import (
    LinfPGDAttack, 
    CarliniWagnerL2Attack, 
    JacobianSaliencyMapAttack, 
    LBFGSAttack
)

map_config_to_attack = {}


map_config_to_attack["CarliniWagnerL2Attack"] = CarliniWagnerL2Attack(model, num_classes = len(classes) , 
	confidence=0, targeted=False, learning_rate=0.01, binary_search_steps=9, max_iterations=10000, 
	abort_early=True, initial_const=0.001, clip_min=0.0, clip_max=1.0, loss_fn=None)

map_config_to_attack["JacobianSaliencyMapAttack"] = JacobianSaliencyMapAttack(model, num_classes = len(classes), 
	clip_min=0.0, clip_max=1.0, loss_fn=None, theta=1.0, gamma=1.0, comply_cleverhans=False)

map_config_to_attack["LinfPGDAttack"] = LinfPGDAttack(
	model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
	nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
	targeted=False)

map_config_to_attack["LBFGSAttack"] = LBFGSAttack(model, num_classes = len(classes), batch_size=1, binary_search_steps=9, 
	max_iterations=100, initial_const=0.01, clip_min=0, clip_max=1, loss_fn=None, targeted=False)
