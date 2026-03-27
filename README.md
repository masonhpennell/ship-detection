# ship-detection
Created by Mason Pennell, Spencer Lafferty, and Jack de Bruyn

Dataset: [Classification of Ship Images](https://www.kaggle.com/code/teeyee314/classification-of-ship-images/input)

## Training modes

`model_pipeline/base_pipeline.py` now supports:

- `MODEL_TYPE = "cnn"`: custom CNN head on EfficientNetB0
- `MODEL_TYPE = "vit"`: custom ViT implementation
- `MODEL_TYPE = "transfer"`: transfer learning with pretrained backbones and two-stage fine-tuning

### Transfer learning config

Set these constants in `base_pipeline.py`:

- `TRANSFER_BACKBONE`: `"MobileNetV2"`, `"EfficientNetB0"`, or `"ResNet50"`
- `TRANSFER_WEIGHTS`: usually `"imagenet"`
- `EPOCHS`: initial feature-extractor training epochs (frozen backbone)
- `FINE_TUNE_EPOCHS`: additional epochs after unfreezing part of backbone
- `UNFREEZE_AT`: layer index where fine-tuning begins
