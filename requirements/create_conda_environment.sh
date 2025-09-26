# create conda environment with all QuOpt-packages in it
yes | conda create -n quopt python=3.10 spyder
source activate quopt
yes | pip install -r requirements.txt
source deactivate