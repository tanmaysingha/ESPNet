# Efficient Segmentation Pyramid Network (ESPNet) 
This reposiotory is for "ESPNet", an efficient scalable scene segmentation model which uses Pyramid Pooling Module (PPM) as a global contextual prior for feature extraction. By exploiting the scalable feature of EfficientNet models, we designed Base ESPNet S0, S1 and S2. We noticed that due to scaling ESPNet from S0 to S2, validation mIoU has increased by 4.4%. But scaling process also increases computational cost due to increase of number of parameters and FLOPS. That is why; instead of scaling up further, we introduced a final ESPNet. Detail description of the models will be available upon acceptance. Few results are provided in the "Results" folders.  
# Requirements for Project
TensorFlow 2.1 (Mainly testing out its capabilities at this time)
This requires CUDA >= 10.1
TensorRT will require an NVIDIA GPU with Tensor Cores (Project will use different GPUs)
On the Jetson AGX Xavier its TensorRT version 6.
Python >= 3.7
