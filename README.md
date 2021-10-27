# Efficient Segmentation Pyramid Network
Semantic or Pixel-wise Segmentation is the process of automatically applying a class or label, designated by a dataset, to each pixel in an image.These labels or classes could include people, car, flower, building, furniture, and etc. To efficiently apply a class or label to each pixel of an image, we introduce an efficient and scalable network, call "Efficient Segmentation Pyramid Network (ESPNet)". By exploiting the scalable feature of EfficientNet models, we design Base ESPNet S0, S1 and S2. We noticed that due to scaling ESPNet from S0 to S2, validation mIoU has increased by 4.4%. But scaling process also increases computational cost due to increase of number of parameters and FLOPS. That is why; instead of scaling up further, we introduce final ESPNet. Model Predictions are uploaded in this reposiotory. 

### Complete pipeline of ESPNet
![pipeline](https://github.com/tanmaysingha/ESPNet/blob/master/Results/ESPNet_pipeline.png?raw=true)

## Datasets
For this research work, we have used cityscapes benchmark datasets.
* Cityscapes - To access this benchmark, user needs an account. https://www.cityscapes-dataset.com/downloads/              

## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Transfer Learning
For performance comparison, we trained few off-line and real-time existing models under same configuration and compared their performance with ESPNet. Some existing models require the use of ImageNet pretrained models to initialize their weights. Details will be given soon.

## Requirements for Project
* TensorFlow 2.1
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores.
  * Horovod framework (for effective utilization of resources and speed up GPUs)
* Keras 2.3.1
* Python >= 3.7

## Results
Size of image in cityscapes dataset is 1024x2048px. But ESPNets accept 512x512px input. Therefore, all models are trained with 512x512px size of input and predicted images are of same size. 
### Separable UNet
![Separable UNet](https://github.com/tanmaysingha/ESPNet/blob/master/Results/other_models/Separable_UNet.png?raw=true)

### DeepLabV3+
![DeepLabV3+](https://github.com/tanmaysingha/ESPNet/blob/master/Results/other_models/DeepLabV3%2B.png?raw=true)

### Bayesian SegNet
![Bayesian SegNet](https://github.com/tanmaysingha/ESPNet/blob/master/Results/other_models/Bayesian_SegNet.png?raw=true)

### FAST-SCNN
![FAST-SCNN](https://github.com/tanmaysingha/ESPNet/blob/master/Results/other_models/FAST_SCNN.png?raw=true)

### Base ESPNet S0
![Base ESPNet](https://github.com/tanmaysingha/ESPNet/blob/master/Results/Base_ESPNet_Results/Base_ESPNet.png?raw=true)

### Final ESPNet
![Final ESPNet](https://github.com/tanmaysingha/ESPNet/blob/master/Results/Final_ESPNet_Results/result1.png?raw=true)

### Citation
 ```yaml
cff-version: 1.2.0
If FANet is useful for your research work, please consider for citing the paper:
@inproceedings{singha2020efficient,
  title={Efficient Segmentation Pyramid Network},
  author={Singha, Tanmay and Pham, Duc-Son and Krishna, Aneesh and Dunstan, Joel},
  booktitle={Proceedings of the International Conference on Neural Information Processing},
  pages={386--393},
  year={2020},
  organization={Springer}
}
```
Refer the following link for the paper:
https://link.springer.com/chapter/10.1007/978-3-030-63820-7_44
