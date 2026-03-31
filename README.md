# Ship Detection with Computer Vision
Created by Mason Pennell, Spencer Lafferty, and Jack de Bruyn

Dataset: [Classification of Ship Images](https://www.kaggle.com/code/teeyee314/classification-of-ship-images/input)

## Why are we doing this?

Ship or vessel detection can be used by a variety of customers, such as the coast gaurd, fishing companies, and military units. This project serves as a way to categorize ships by their type

1. Cargo
2. Military
3. Carrier
4. Cruise
5. Tankers

This project uses the machine learning library TensorFlow because creating a CV system like this requires deep learning. Object detection and tracking are important for modern maritime duties, ensuring real-time monitoring and analysis of the ocean by discerning ships from other obstacles. These systems help crews conduct long-term maritime surveillance, minimize human errors, and support remote-controlled and autonomous ships. They enable rapid and accurate localization of objects, guide rescue operations, monitor key areas, track suspicious ships, and combat illegal activities. Therefore, the development of shipborne visual object detection and tracking technology has profound impact on promoting the advancement of navigational technology. However, due to the complexity of the environment, ship detection and tracking remain meaningful yet challenging tasks.

## Instructions

- Requires python 3.10+
- Follow instructions for [TensorFlow](https://www.tensorflow.org/install)

### Transfer learning config

Set these constants in `base_pipeline.py`:

- `MODEL_TYPE`: change the model between `CNN`, `ViT`, or `transfer` for pretrained weights
- `TRANSFER_BACKBONE`: `"MobileNetV2"`, `"EfficientNetB0"`, or `"ResNet50"`
- `TRANSFER_WEIGHTS`: usually `"imagenet"`
- `EPOCHS`: initial feature-extractor training epochs (frozen backbone)
- `FINE_TUNE_EPOCHS`: additional epochs after unfreezing part of backbone
- `UNFREEZE_AT`: layer index where fine-tuning begins
