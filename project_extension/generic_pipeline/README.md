# Some Example Executions :

python attack_launcher.py --attack LBFGSAttack --modelfile ./densenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/densenet_BFGS


python attack_launcher.py --attack LBFGSAttack --modelfile ./mobilenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/mobilenet_BFGS

python attack_launcher.py --attack LBFGSAttack --modelfile ./squeezenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/squeezenet_BFGS

python attack_launcher.py --attack LBFGSAttack --modelfile ./mnasnet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/mnasnet_BFGS


python attack_launcher.py --attack GradientSignAttack --modelfile ./mobilenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/mobilenet_GSA

python attack_launcher.py --attack GradientSignAttack --modelfile ./densenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/densenet_GSA

python attack_launcher.py --attack GradientSignAttack --modelfile ./squeezenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/squeezenet_GSA

python attack_launcher.py --attack GradientSignAttack --modelfile ./alexnet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/alexnet_GSA


python attack_launcher.py --attack ElasticNetL1Attack --modelfile ./alexnet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/alexnet_ENL1A




python attack_launcher.py --attack DDNL2Attack --modelfile ./mobilenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/mobilenet_DDLN2A

python attack_launcher.py --attack DDNL2Attack --modelfile ./densenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/densenet_DDLN2A

python attack_launcher.py --attack DDNL2Attack --modelfile ./squeezenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/squeezenet_DDLN2A

python attack_launcher.py --attack DDNL2Attack --modelfile ./alexnet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/alexnet_DDLN2A

python attack_launcher.py --attack DDNL2Attack --modelfile ./densenet_tl/modelfile.pth  --num_folds 1 --batch_size 50 --outdir small_batch/densenet_DDLN2A

