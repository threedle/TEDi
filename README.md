# TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis

[[Project Page](https://threedle.github.io/TEDi/)] [[ArXiv](https://arxiv.org/abs/2307.15042)]


## Installation

```
conda env create -f environment.yml
conda activate tedi
```

## Usage

### Training
(Data download coming soon...)

To train on the CMU dataset with the default set of hyperparameters described in ```src/util/options.py``` run

```
cd src
python train_progressive_diffusion.py
```

### Testing
Coming soon...


### Custom dataset
Coming soon...

## License
The project is distributed under the [MIT License](https://github.com/GuyTevet/motion-diffusion-model/blob/main/LICENSE)

## Acknowledgements
The code is based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
