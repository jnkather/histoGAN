# histoGAN: Generative adversarial networks for histology images

By the Kather lab: www.kather.ai

## What is this?
This is a Matlab R2020a-based tool to train and use GANs for histological images

## How to use?

### Find an image data set
First, you need to find an image data set to train a GAN on. For example, you can use the training images in this repository: https://zenodo.org/record/3832231 

### Train the GAN
Now, you can train the GAN on this image set, like this:

```
runCGAN('inPx',512,'miniBatchSize',16,'gpuDev',2,...
'inputDir','<PATH_TO_TRAINING_IMAGES>',...
'flipFactor',0.50,'discriminatorBnEps',5e-4,'generatorBnEps',5e-4,'alternateNet',3)
```

### Deploy the GAN
Next, you can use a trained GAN (for example, this ###) and use it to generate synthetic images, like this:

```
deployCGAN('miniBatchSize',64,'gpuDev',1,'relPathTrainedModel','RYIKDFCWQE/chkpt_00300000.mat','numGenerate',100000)
```
