import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers

# =========================================================
# CONFIGURATION
# =========================================================
class Config:
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 8
    EPOCHS = 25
    DATASET_PATH = "dataset"
    IMAGE_DIR = os.path.join(DATASET_PATH, "images")
    MASK_DIR = os.path.join(DATASET_PATH, "masks")
    VAL_SPLIT = 0.2
    AUTOTUNE = tf.data.AUTOTUNE
    NUM_CLASSES = 1  # Binary segmentation
    LEARNING_RATE = 1e-4
    SEED = 42
    MODEL_PATH = "unet_model.h5"


# =========================================================
# DATA LOADING
# =========================================================
def get_image_mask_pairs(image_dir, mask_dir):
    image_exts = (".jpg", ".jpeg", ".png")
    mask_exts = (".png", ".jpg", ".jpeg")

    image_paths = sorted([
        p for p in glob.glob(os.path.join(image_dir, "*"))
        if p.lower().endswith(image_exts)
    ])

    # Build mask lookup by basename
    mask_files = glob.glob(os.path.join(mask_dir, "*"))
    mask_dict = {}

    for m in mask_files:
        base = os.path.splitext(os.path.basename(m))[0]
        mask_dict[base] = m

    pairs = []

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]

        if base not in mask_dict:
            print(f"WARNING: No mask found for {base}")
            continue

        pairs.append((img_path, mask_dict[base]))

    if len(pairs) == 0:
        raise ValueError("No valid image-mask pairs found.")

    print(f"✅ Found {len(pairs)} valid image-mask pairs")

    images, masks = zip(*pairs)
    return list(images), list(masks)


def train_val_split(image_paths, mask_paths, val_split):
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)

    split_idx = int(len(combined) * (1 - val_split))
    train = combined[:split_idx]
    val = combined[split_idx:]

    train_images, train_masks = zip(*train)
    val_images, val_masks = zip(*val)

    return list(train_images), list(train_masks), list(val_images), list(val_masks)


# =========================================================
# PREPROCESSING
# =========================================================
def parse_image(img_path, mask_path):
    # --- IMAGE ---
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # ✅ FIXED
    img.set_shape([None, None, 3])  # explicitly set shape

    img = tf.image.resize(img, Config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    # --- MASK ---
    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)  # ✅ FIXED
    mask.set_shape([None, None, 1])

    mask = tf.image.resize(mask, Config.IMAGE_SIZE, method="nearest")
    mask = tf.cast(mask > 0, tf.float32)

    return img, mask


# =========================================================
# AUGMENTATION
# =========================================================
def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        mask = tf.image.rot90(mask, k)

    return img, mask


# =========================================================
# TF.DATA PIPELINE
# =========================================================
def build_dataset(image_paths, mask_paths, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    dataset = dataset.map(
        lambda x, y: parse_image(x, y),
        num_parallel_calls=Config.AUTOTUNE
    )

    if training:
        dataset = dataset.map(augment, num_parallel_calls=Config.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()   # ✅ FIX
    else:
        dataset = dataset.cache()

    dataset = dataset.batch(Config.BATCH_SIZE)
    dataset = dataset.prefetch(Config.AUTOTUNE)

    return dataset


# =========================================================
# MODEL (U-NET WITH MOBILENETV2 ENCODER)
# =========================================================
def unet_model(input_shape=(256, 256, 3)):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Encoder layers
    layer_names = [
        "block_1_expand_relu",   # 128x128
        "block_3_expand_relu",   # 64x64
        "block_6_expand_relu",   # 32x32
        "block_13_expand_relu",  # 16x16
        "block_16_project",      # 8x8
    ]

    layers_outputs = [base_model.get_layer(name).output for name in layer_names]
    encoder = keras.Model(inputs=base_model.input, outputs=layers_outputs)

    encoder.trainable = False  # freeze initially

    inputs = keras.Input(shape=input_shape)
    skips = encoder(inputs)
    x = skips[-1]

    # Decoder
    decoder_filters = [512, 256, 128, 64]

    for i in range(4):
        x = layers.Conv2DTranspose(decoder_filters[i], 3, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[-(i + 2)]])
        x = layers.Conv2D(decoder_filters[i], 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(decoder_filters[i], 3, activation="relu", padding="same")(x)

    # Final upsampling to match input resolution (256x256)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


# =========================================================
# METRICS
# =========================================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def iou_score(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


# =========================================================
# VISUALIZATION
# =========================================================
def display_predictions(model, dataset, num=3):
    for images, masks in dataset.take(1):
        preds = model.predict(images)

        for i in range(num):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(images[i])
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(tf.squeeze(masks[i]), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(tf.squeeze(preds[i] > 0.5), cmap="gray")
            plt.axis("off")

            plt.show()


# =========================================================
# INFERENCE
# =========================================================
def predict_single_image(model, image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, Config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    input_img = tf.expand_dims(img, axis=0)
    pred = model.predict(input_img)[0]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(tf.squeeze(pred > 0.5), cmap="gray")
    plt.axis("off")

    plt.show()


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    tf.random.set_seed(Config.SEED)
    random.seed(Config.SEED)

    print("Loading dataset...")
    image_paths, mask_paths = get_image_mask_pairs(
        Config.IMAGE_DIR, Config.MASK_DIR
    )

    train_imgs, train_masks, val_imgs, val_masks = train_val_split(
        image_paths, mask_paths, Config.VAL_SPLIT
    )

    train_ds = build_dataset(train_imgs, train_masks, training=True)
    val_ds = build_dataset(val_imgs, val_masks, training=False)

    print("Building model...")
    model = unet_model(input_shape=(*Config.IMAGE_SIZE, 3))

    model.compile(
        optimizer=keras.optimizers.Adam(Config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[dice_coefficient, iou_score]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            Config.MODEL_PATH, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True
        )
    ]

    print("Training...")

    validation_steps = math.ceil(len(val_imgs) / Config.BATCH_SIZE)
    steps_per_epoch = math.ceil(len(train_imgs) / Config.BATCH_SIZE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    print("Evaluating...")
    model.evaluate(val_ds)

    print("Displaying predictions...")
    display_predictions(model, val_ds)

    # Example inference (replace with real path)
    # predict_single_image(model, "dataset/images/example.jpg")


if __name__ == "__main__":
    main()