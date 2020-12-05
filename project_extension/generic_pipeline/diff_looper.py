

import os
from tqdm import tqdm 

def create_dir(name):
	if not os.path.exists(name):
		os.makedirs(name)

main_directory = os.getcwd()

script_path = os.path.join(main_directory, "../utils/diff_DCT.py")
script_path = script_path.replace(' ', '\\ ')

#script_path = "../../../../../utils/diff_DCT.py"

#print(script_path)

attacks = ['PGD', 'DDNL', 'GSA']
models = ['alexnet', 'squeezenet', 'densenet', 'mnasnet', 'mobilenet']


# model_wise accross_attacks

model_wise_folder = os.path.join(main_directory, "DCT_DIFFs_model_wise")
create_dir(model_wise_folder)

#print(model_wise_folder)

print("\n\n MODEL-WISE : ")
for model in tqdm(models):

	model_path = os.path.join(model_wise_folder, model)

	for attack1 in tqdm(attacks):
		for attack2 in tqdm(attacks):
			if attack1==attack2:
				continue

			foldername = f"{model}_{attack1}_vs_{attack2}"
			foldername = os.path.join(model_path, foldername)

			create_dir(foldername)

			targeted_folder = os.path.join(foldername, "targeted")

			first_png_path = f"./small_batch/{model}_e100{attack1}/logscale_correct_DCT_targeted.png"
			second_png_path = f"./small_batch/{model}_e100{attack2}/logscale_correct_DCT_targeted.png"

			first_png_path = os.path.join(main_directory, first_png_path).replace(' ', '\\ ')
			second_png_path = os.path.join(main_directory, second_png_path).replace(' ', '\\ ')

			create_dir(targeted_folder)
			os.chdir(targeted_folder)
			os.system(f'python {script_path} --first {first_png_path}  --second {second_png_path} -a {attack1} -b {attack2}')

			# switch back to main dir
			os.chdir(main_directory)
			

			untargeted_folder = os.path.join(foldername, "untargeted")

			first_png_path = f"./small_batch/{model}_e100{attack1}/logscale_correct_DCT_untargeted.png"
			second_png_path = f"./small_batch/{model}_e100{attack2}/logscale_correct_DCT_untargeted.png"

			first_png_path = os.path.join(main_directory, first_png_path).replace(' ', '\\ ')
			second_png_path = os.path.join(main_directory, second_png_path).replace(' ', '\\ ')

			create_dir(untargeted_folder)
			os.chdir(untargeted_folder)

			os.system(f'python {script_path} --first {first_png_path}  --second {second_png_path} -a {attack1} -b {attack2}')


			# switch back to main dir
			os.chdir(main_directory)


attack_wise_folder = os.path.join(main_directory, "DCT_DIFFs_attack_wise")
create_dir(attack_wise_folder)



print("\n\n ATTACK-WISE : ")
for attack in tqdm(attacks):

	attack_path = os.path.join(attack_wise_folder, attack)

	for model1 in tqdm(models):
		for model2 in tqdm(models):
			if model1==model2:
				continue

			foldername = f"{attack}_{model1}_vs_{model2}"
			foldername = os.path.join(attack_path, foldername)

			create_dir(foldername)

			targeted_folder = os.path.join(foldername, "targeted")

			create_dir(targeted_folder)

			first_png_path = f"./small_batch/{model1}_e100{attack}/logscale_correct_DCT_targeted.png"
			second_png_path = f"./small_batch/{model2}_e100{attack}/logscale_correct_DCT_targeted.png"

			first_png_path = os.path.join(main_directory, first_png_path).replace(' ', '\\ ')
			second_png_path = os.path.join(main_directory, second_png_path).replace(' ', '\\ ')

			os.chdir(targeted_folder)
			os.system(f'python {script_path} --first {first_png_path}  --second {second_png_path} -a {model1} -b {model2}')

			untargeted_folder = os.path.join(foldername, "untargeted")

			create_dir(untargeted_folder)

			first_png_path = f"./small_batch/{model1}_e100{attack}/logscale_correct_DCT_untargeted.png"
			second_png_path = f"./small_batch/{model2}_e100{attack}/logscale_correct_DCT_untargeted.png"

			first_png_path = os.path.join(main_directory, first_png_path).replace(' ', '\\ ')
			second_png_path = os.path.join(main_directory, second_png_path).replace(' ', '\\ ')

			os.chdir(untargeted_folder)
			os.system(f'python {script_path} --first {first_png_path}  --second {second_png_path} -a {model1} -b {model2}')


			# switch back to main dir
			os.chdir(main_directory)



target_vs_untarget_folder = os.path.join(main_directory, "DCT_DIFFs_TARGET_vs_UNTARGET")
create_dir(target_vs_untarget_folder)


print("\n\n TARGETED_vs_UNTARGETED : ")

for attack in tqdm(attacks):

	attack_path = os.path.join(target_vs_untarget_folder, attack)

	for model in tqdm(models):
	
		foldername = f"{attack}_{model}_target_vs_untarget"
		foldername = os.path.join(attack_path, foldername)

		create_dir(foldername)
		os.chdir(foldername)

		first_png_path = f"./small_batch/{model}_e100{attack}/logscale_correct_DCT_targeted.png"
		second_png_path = f"./small_batch/{model}_e100{attack}/logscale_correct_DCT_untargeted.png"

		first_png_path = os.path.join(main_directory, first_png_path).replace(' ', '\\ ')
		second_png_path = os.path.join(main_directory, second_png_path).replace(' ', '\\ ')

		os.system(f'python {script_path} --first {first_png_path}  --second {second_png_path} -a targeted -b untargeted')

		# switch back to main dir
		os.chdir(main_directory)

