# mosgim

## Install 

Create virtual environment using your favorite tool. Below is anaconda example: 

    conda create -n mosgim_service python=3.10
    conda deactivate 
    conda activate mosgim_service

Install `poetry`:

    pip install poetry 
    
Install `mosgim`, in project root run bash command:
    
    poetry install 

## Use test data

https://cloud.iszf.irk.ru/index.php/s/AMynoe9RzCC3KD6


## Run processing

Use `process.py` script in `scripts` folder:

python process.py --data_path /PATH/TO/DATA/ --process_type single --data_source txt --date 2017-01-02 --ndays 1 --mag_type mdip --nworkers 7

    * `/PATH/TO/DATA/` should contain sites data from `2017_002.zip` 
    * Adjust `nworkers` up to CPU cores you are able to use for computations.