# Efficient Segmentation Pyramid Network
Semantic or Pixel-wise Segmentation is the process of automatically applying a class or label, designated by a dataset, to each pixel in an image.These labels or classes could include people, car, flower, building, furniture, and etc. To efficiently apply a class or label to each pixel of an image, we introduce an efficient and scalable network, call "Efficient Segmentation Pyramid Network (ESPNet)". By exploiting the scalable feature of EfficientNet models, we design Base ESPNet S0, S1 and S2. We noticed that due to scaling ESPNet from S0 to S2, validation mIoU has increased by 4.4%. But scaling process also increases computational cost due to increase of number of parameters and FLOPS. That is why; instead of scaling up further, we introduce final ESPNet. Model Predictions are uploaded in this reposiotory. Details will be available upon acceptance of research paper.

## Datasets
For this research work, we have used cityscapes benchmark datasets.
* Cityscapes - To access this benchmark, user needs an account. https://www.cityscapes-dataset.com/downloads/              

## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Transfer Learning
For performance comparison, we trained few off-line and real-time existing models under same configuration and compared their performance with ESPNet. Some existing models require the use of ImageNet pretrained models to initialize their weights. DEtails will be given soon.

## Requirements for Project
* TensorFlow 2.1
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores.
* Keras 2.3.1
* Python >= 3.7

## Results
Size of image in cityscapes dataset is 1024x2048px. But ESPNets accept 512x512px input. Therefore, all models are trained with 512x512px size of input and results are generated of same size. 
### Separable UNet


### DeepLabV3+


### Bayesian SegNet


### FAST-SCNN


### Final ESPNet
![Final ESPNet](https://github.com/tanmaysingha/ESPNet/blob/master/Results/Final_ESPNet_Results/result1.png?raw=true)

### Base ESPNet
![Base ESPNet](https://github.com/tanmaysingha/ESPNet/blob/master/Results/Final_ESPNet_Results/result2.png?raw=true)
