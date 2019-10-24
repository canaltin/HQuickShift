# HQuickShift
The supplementary material for Hierarchical Quick Shift Guided Recurrent Clustering Algorithm [ICDE 2020]

An implementation including the datasets, 
the trained RNN models, the jupyter notebook to reproduce
all figures and our experiment results with corresponding
explanations related to software and hardware setup.

Installed Programs: 
------------------------------------------------------------------------------------
Install Anaconda or Miniconda by following the instructions 
[https://conda.io/docs/user-guide/install/linux.html].

You can find a short summary below:
Installing Anaconda or Miniconda on Linux
Download the installer:
    Miniconda installer for Linux https://conda.io/miniconda.html
    Anaconda installer for Linux https://www.anaconda.com/download/
In your Terminal window, run:
Miniconda:
.. code::
bash Miniconda3-latest-Linux-x86_64.sh
Anaconda:
.. code::
bash Anaconda-latest-Linux-x86_64.sh

Follow the prompts on the installer screens.
If you are unsure about any setting, accept the defaults. You
can change them later.

To make the changes take effect, close and then re-open your Terminal window.

Test your installation
To test your installation, in your Terminal window or Anaconda Prompt,
run the command conda list.
For a successful installation, a list of installed packages appears.

####################################################################################

After installing Anaconda or Miniconda you can create a conda environment 
following the steps creating an environment from the provided environment.yml file:
    Use the terminal or an Anaconda Prompt for the following steps:
       - Create the environment from the environment.yml file:
         conda env create -f environment.yml
       - Activate the new environment: 
         conda activate hqshift

After activating the environment, you can run the jupyter notebook: 
"icde_2019_372.ipynb" after running the command from terminal: 'jupyter notebook'.
Html file is provided as a noninteractive documentation.
 
With this notebook, you can able to reproduce the results presented in the paper.

Furthermore,
    datasets are located under 'data_sets' folder, 
    trained models are located under 'models' folder. 

Folder/File Description:
- data_sets: the real datasets used with corresponding scripts to download/preprocess/
and run the batch algorithms.
- models: trained RNN models. These models can be directly loaded and tested, see
icde_2019_372.ipynb.
- environment.yml: To recreate the working environment to reproduce the results.
- icde_2019_372.ipynb: The main notebook to reproduce the results presented in paper.
- icde_2019_372.html: A noninteractive version of the provided jupyter notebook.
- README: this file.
