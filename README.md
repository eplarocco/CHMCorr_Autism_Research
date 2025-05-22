# CHMCorr_Autism_Research
Data Preprocessing and CHM Corr Algorithm Prototype for Structural MRI Images

## Important files and what they do
### CHMCorr_Autism_Research/01_Data_Preprocessing.ipynb

Follows the code in -  Towards 3D Deep Learning for neuropsychiatry: predicting Autism diagnosis using an interpretable Deep Learning pipeline applied to minimally processed structural MRI data, Melanie Garcia, Clare Kelly. medRxiv 2022.10.18.22281196; doi: https://doi.org/10.1101/2022.10.18.22281196

Github: https://github.com/garciaml/Autism-3D-CNN-brain-sMRI?tab=readme-ov-file

Inputs: raw ABIDEI, ABIDEII, and ACE datasets - Saved in Original_Data folder (also houses Metadata)
1. Converts them into BIDS folder structure (https://bids.neuroimaging.io/) - Saved in Restructured_Data folder
2. Removes non-brain tissue via BET (https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/bet) - Saved in JustBrain_Data folder (also houses participants.tsv file)
3. Preprocesses the data accoring to the steps in Garcia et al. - Saved in Preprocessed_Data folder

### CHMCorr_Autism_Research/02_Finetune_Resnet.ipynb (can just use command line)

Combines ABIDEI and ABIDEII datasets for training

#### Start training from pre-trained Medical ResNet50

python ../Autism-3D-CNN-brain-sMRI/train_medicalnet.py 'JustBrain_Data/ABIDE_COMBINED' 'Preprocessed_Data/ABIDE_COMBINED' './outputs/Resnet50/ABIDE_Combined' '../Autism-3D-CNN-brain-sMRI/resnet_training/resnet_10.pth' --lr 0.0003 --batch 8 --epochs 10


#### Evaluate on Test Set (swap checkpoint_X.pth) for whichever epoch performed the best

python ../Autism-3D-CNN-brain-sMRI/predict_medicalnet_subids.py 'Preprocessed_Data/ABIDE_COMBINED/test' 'outputs/Resnet50/ABIDE_Combined/test/subjects.csv' './outputs/Resnet50/ABIDE_Combined/checkpoint_18.pth' './outputs/Resnet50/ABIDE_Combined/test'

# Relies on eplarocco/Autism-3D-CNN-brain-sMRI repo (including saved resnet model path - must upload manually) and folders in this repo: JustBrain_Data and Preprocessed_Data - saves results to outputs/Resnet50/