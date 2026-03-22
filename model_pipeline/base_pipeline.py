import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import os

import kagglehub

#load data
path = kagglehub.dataset_download("arpitjain007/game-of-deep-learning-ship-datasets")

TRAIN_DIR = os.path.join(path, "train")
TEST_DIR  = os.path.join(path, "test")

#config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 15
MODEL_TYPE = "cnn"  # "cnn" or "vit"

#preprocessing
def load_datasets(train_dir, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

#normalization
normalization_layer = layers.Rescaling(1./255)

#data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def prepare(ds, training=False):
    ds = ds.map(lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)

#model architectures, both CNN and Vit
#CNN
def build_cnn_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )

    base_model.trainable = True  # fine-tune later if needed

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=base_model.input, outputs=outputs)

#ViT
class Patches(layers.Layer):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=tf.shape(patches)[1], delta=1)
        return self.projection(patches) + self.position_embedding(positions)
    
def build_vit_model(num_classes):
    patch_size = 16
    num_patches = (IMG_SIZE[0] // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_layers = 6

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([attention, encoded])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dense(projection_dim)(x3)

        encoded = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(encoded)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

#training
def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=3),
    keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

train_ds, val_ds, class_names = load_datasets(TRAIN_DIR, IMG_SIZE, BATCH_SIZE)

train_ds = prepare(train_ds, training=True)
val_ds   = prepare(val_ds, training=False)

num_classes = len(class_names)

if MODEL_TYPE == "cnn":
    model = build_cnn_model(num_classes)
else:
    model = build_vit_model(num_classes)

model = compile_model(model)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

#eval
loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.4f}")

#test predictions
def predict_batch(model, dataset):
    preds = model.predict(dataset)
    return tf.argmax(preds, axis=1)

def predict_image(model, img_path, class_names):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = normalization_layer(img)
    img = tf.expand_dims(img, axis=0)

    preds = model.predict(img)
    return class_names[np.argmax(preds)]