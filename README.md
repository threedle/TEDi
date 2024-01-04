# TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis

[[Project Page](https://threedle.github.io/TEDi/)]


## Installation

First clone the main branch and create the envrionment by running

```
conda env create -f environment.yml
conda activate tedi
```

## Usage

### Training
To train from scratch on the CMU dataset, first download the data from this [Google Drive link](https://drive.google.com/file/d/1lYNBCkdYZwGeOoyO2MKkiWDR-87xAYpU/view?usp=sharing). (_Note: this is a very densely sampled version standing at 34G uncompressed unfortunately, see [custom dataset](#custom-dataset) if you want to use your own set of BVH files_). Please extract the downloaded file and place it under a new directory  ```data/processed/```

Then, to train with the default set of hyperparameters described in ```src/util/options.py``` (you might want to replace the default ```data_path``` option with the actual path of your extracted data) run

```
cd src
python train_progressive_diffusion.py
```
(_Note: Training is long and can easily take more than 24 hours depending on the GPU, but it does support training with [accelerate](https://huggingface.co/docs/accelerate/) credit to [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)_)

### Testing
To generate long sequences conditioning on a primer, you need to download the CMU data as in the previous section (the primers will be randomly selected motion sequences), and then download a pretrained model from this [Google Drive link](https://drive.google.com/file/d/17_Eqp1pScQuB8sSomBmWOL9GzKtT7b-9/view?usp=sharing). Please extract the downloaded file and place it under a new directory ```exps/``` under the project directory.

Then, run 
```
cd src
python test_progressive_diffusion.py --exp_names {path to pretrained model directory} --n_samples {number of generated motions} --sample_len {number of frames}
```
 (_Note: primers for CMU data is 500 frames, so ```sample_len``` should be greater than that_). 

### Custom dataset
Coming soon...

## License
The project is distributed under the [MIT License](https://github.com/GuyTevet/motion-diffusion-model/blob/main/LICENSE)

## Acknowledgements
The code is based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
