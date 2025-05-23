

vqgan_ckpt=pretrained_models/AutoencoderModel.ckpt
datapath="/mnt/realccvl15/yyang303/registered/"
tumorlabel="/mnt/realccvl15/yyang303/registered/"
# tumorlabel="/ccvl/net/ccvl15/xinran/Tumor/pancreas/"
# tumorlabel="/ccvl/net/ccvl15/xinran/Tumor/kidney/"
 
python3 train.py dataset.name=liver_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['liver'] dataset.uniform_sample=False model.results_folder_postfix="liver" model.vqgan_ckpt=$vqgan_ckpt
# python3 train.py dataset.name=pancreas_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['pancreas'] dataset.uniform_sample=False model.results_folder_postfix="pancreas" model.vqgan_ckpt=$vqgan_ckpt
# python3 train.py dataset.name=kidney_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['kidney'] dataset.uniform_sample=False model.results_folder_postfix="kidney" model.vqgan_ckpt=$vqgan_ckpt

# sbatch --error=logs/diffusion_model.out --output=logs/diffusion_model.out hg.sh
