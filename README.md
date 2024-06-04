## Self-supervised learning with chest X-ray images and electronic health records data

Table of contents
=================

<!--ts-->
  * [Background](#Background)
  * [Environment setup](#Environment-setup)
  * [Model training](#Model-training)
  * [Model evaluation](#Model-evaluation)
  * [Citation](#Citation)
   
<!--te-->

Background
============
We follow the data extraction and linking pipeline of the two datasets MIMIC-IV and MIMIC-CXR based on the task definition (i.e., inhospital mortality prediction,or phenotype classification) using the MedFuse code. 

Environment setup
==================
We originally follow the medfuse environment, however to run this repo, you must install and run a few more libraries that are currently NOT in the below yml file provided by medfuse.

    git clone https://github.com/nyuad-cai/MedFuse.git
    cd MedFuse
    conda env create -f environment.yml
    conda activate medfuse

Note that the code uses neptune.ai for tracking the different training models - HIGHLY RECOMMENDED.

Training and evaluation framework
====================================

Self-supervised pre-training scripts
-----------------

For pre-training, there are three types of training scripts that have been setup for phenotyping and mortality (/task/train/script.sh):
- simclr.sh
- vicreg.sh
- time_simclr.sh

All of the scripts above call run_gpu.py

Note that some of the parameters across all training frameworks are denoted as "simclr" but this is only to indicate the setup of the pre-training architecture. 

For the above, you need to be careful with the naming conventions used to store the model checkpoints. 


Self-supervised evaluation scripts
------------------
To evaluate the quality of the representations learned using pre-training, several scripts have been implemented:
- (task/train/script.sh) finetune.sh, finetune_cxr.sh, finetune_ehr.sh --> these scripts further fine-tune either the multi-modal architecture or the uni-modal branches.
- (task/train/script.sh) lineareval.sh, lineareval_cxr.sh, lineareval_ehr.sh --> these scripts tune a single layer and freeze the pre-trained encoders for either multi-modal or uni-modal predictions.
- (task/eval/script.sh) lineareval / fteval --> these scripts perform an evaluation run for either fine-tuned or linear classifiers.

All of the scripts above call run_gpu.py

Other useful scripts:
- eval_epoch.sh --> This calls epoch_evaluation.py and evaluates the quality of the representations in terms of AUROC using a linear classifier at each pre-training epoch. It is useful for selecting the epoch that yields the best AUROC on the validation set. It stores results in a csv file.
- model_selection_task.sh --> This calls model_selection.py and it is similar to the above (Please compare both scripts to understand the differences between them).

Notebooks
-----------------
There are a few notebooks that may be helpful:
- final_results.ipynb to plot the results and obtain the best epoch.
- cxr_mortality_labels.ipynb, cxr_mortality_labels_2.ipynb, cxr_phenotype_labels.ipynb are used to assign EHR labels to CXR
- dataset.ipynb investigates covariances across embeddings / similarities.
- prototype.ipynb (old notebook for random functions - can be ignored)
- results (SAIL abstract).ipynb was used to generate results for a conference abstract (may be interesting but could also be reformatted for a new submission)
- simclr_auroc_epoch.ipynb notebook that evaluates pre-trained encoders at each epoch (exploratory). 


Citation 
============

If you use this code for your research, please consider citing: TODO
