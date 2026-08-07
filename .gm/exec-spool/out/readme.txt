
# Resemble Enhance

  
    
    

    EnhanceVideo.mp4
    
  

  

  

Resemble Enhance is an AI-powered tool that aims to improve the overall quality of speech by performing denoising and enhancement. It consists of two modules: a denoiser, which separates speech from a noisy audio, and an enhancer, which further boosts the perceptual audio quality by restoring audio distortions and extending the audio bandwidth. The two models are trained on high-quality 44.1kHz speech data that guarantees the enhancement of your speech with high quality.

## Usage

### Installation
Install the stable version:

```
pip install resemble-enhance --upgrade
```

Or try the latest pre-release version:

```
pip install resemble-enhance --upgrade --pre
```

### Enhance

```
resemble-enhance in_dir out_dir

```

### Denoise only

```
resemble-enhance in_dir out_dir --denoise_only

```

### Web Demo
We provide a web demo built with Gradio, you can try it out here, or also run it locally:

```
python app.py

```

## Train your own model

### Data Preparation
You need to prepare a foreground speech dataset and a background non-speech dataset. In addition, you need to prepare a RIR dataset (examples).

```
data
├── fg
│   ├── 00001.wav
│   └── ...
├── bg
│   ├── 00001.wav
│   └── ...
└── rir
    ├── 00001.npy
    └── ...
```

### Training

#### Denoiser Warmup
Though the denoiser is trained jointly with the enhancer, it is recommended for a warmup training first.

```
python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser
```

#### Enhancer
Then, you can train the enhancer in two stages. The first stage is to train the autoencoder and vocoder. And the second stage is to train the latent conditional flow matching (CFM) model.

##### Stage 1

```
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1
```

##### Stage 2

```
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2
```

## Blog
Learn more on our website!

