# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python (pretrainedresnet2)
#     language: python
#     name: pretrainedresnet2
# ---

# %% [markdown]
# # Citations

# %% [markdown]
# Towards 3D Deep Learning for neuropsychiatry: predicting Autism diagnosis using an interpretable Deep Learning pipeline applied to minimally processed structural MRI data, Melanie Garcia, Clare Kelly. medRxiv 2022.10.18.22281196; doi: https://doi.org/10.1101/2022.10.18.22281196
#
# Github: https://github.com/garciaml/Autism-3D-CNN-brain-sMRI?tab=readme-ov-file

# %% [markdown]
# # Virtual Environment

# %%
# Activate Virtual Environment and Install Requirements
# #!python3 -m venv ../pretrainedresnet2
# #!source ../pretrainedresnet2/bin/activate
# #!python3 -m ipykernel install --user --name=pretrainedresnet2 --display-name "Python (pretrainedresnet2)"
#Switch to notebook/virtual environment kernel

# %% [markdown]
# # Libraries and Imports

# %%
# #!pip install -r requirements.txt
# #!pip install "torchio>=0.19.0"
# #!pip install monai
# #!pip install tensorboard
# #!pip install torchsummary
import pandas as pd

# %% [markdown]
# # Finetune Model from Pretrained Medical ResNet50

# %%
# Start training from scratch
# !python ../Autism-3D-CNN-brain-sMRI/train_medicalnet.py 'JustBrain_Data/ABIDE_COMBINED' 'Cleaned_Data/ABIDE_COMBINED' './outputs/Resnet50/ABIDE_COMBINED' '../Autism-3D-CNN-brain-sMRI/resnet_training/resnet_18_23dataset.pth' --lr 0.0003 --batch 8 --epochs 20

# %%
# Resume training from saved model
# #!python ../Autism-3D-CNN-brain-sMRI/train_medicalnet.py 'JustBrain_Data/ABIDE_COMBINED' 'Cleaned_Data/ABIDE_COMBINED' './outputs/Resnet50/ABIDE_COMBINED' 'outputs/Resnet50/ABIDE_COMBINED/checkpoint_7.pth' --resume

# %% [markdown]
# ## Test

# %%
participants_tsv = pd.read_csv('JustBrain_Data/ABIDE_COMBINED/participants.tsv', sep="\t", dtype=str)
participants_tsv.rename(columns={"participant_id" : "SUB_ID"}, inplace=True)
participants_tsv = participants_tsv[participants_tsv.dataset == 'test']
participants_tsv.to_csv('outputs/Resnet50/ABIDE_COMBINED/test/subjects.csv', index = False)

# %%
# Predictions - 7 epochs (best accuracy)
# !python ../Autism-3D-CNN-brain-sMRI/predict_medicalnet_subids.py 'Cleaned_Data/ABIDE_COMBINED/test' 'outputs/Resnet50/ABIDE_COMBINED/test/subjects.csv' './outputs/Resnet50/ABIDE_COMBINED/checkpoint_7.pth' './outputs/Resnet50/ABIDE_COMBINED/test'

# %% [markdown]
# ## Val

# %%
participants_tsv = pd.read_csv('JustBrain_Data/ABIDE_COMBINED/participants.tsv', sep="\t", dtype=str)
participants_tsv.rename(columns={"participant_id" : "SUB_ID"}, inplace=True)
participants_tsv = participants_tsv[participants_tsv.dataset == 'val']
participants_tsv.to_csv('outputs/Resnet50/ABIDE_COMBINED/validation/subjects.csv', index = False)

# %%
# Predictions - 36 epochs (predictions all 0) - overfit
# !python ../Autism-3D-CNN-brain-sMRI/predict_medicalnet_subids.py 'Preprocessed_Data/ABIDE_COMBINED/val' 'outputs/Resnet50/ABIDE_Combined/validation/subjects.csv' './outputs/Resnet50/ABIDE_Combined/checkpoint_36.pth' './outputs/Resnet50/ABIDE_Combined/validation'

# %% [markdown]
# ## Train

# %%
participants_tsv = pd.read_csv('JustBrain_Data/ABIDE_COMBINED/participants.tsv', sep="\t", dtype=str)
participants_tsv.rename(columns={"participant_id" : "SUB_ID"}, inplace=True)
participants_tsv = participants_tsv[participants_tsv.dataset == 'train']
participants_tsv.to_csv('outputs/Resnet50/ABIDE_COMBINED/train/subjects.csv', index = False)

# %%
# Predictions - 36 epochs (predictions all 0) - overfit
# !python ../Autism-3D-CNN-brain-sMRI/predict_medicalnet_subids.py 'Preprocessed_Data/ABIDE_COMBINED/train' 'outputs/Resnet50/ABIDE_Combined/train/subjects.csv' './outputs/Resnet50/ABIDE_Combined/checkpoint_36.pth' './outputs/Resnet50/ABIDE_Combined/train'
