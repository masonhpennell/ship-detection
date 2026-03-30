from pandas import read_csv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset locations (relative to this file)
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")  # folder containing all images

#config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 15
MODEL_TYPE = "transfer"  # "cnn", "vit", or "transfer"
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5

# transfer learning config
TRANSFER_WEIGHTS = "imagenet"
TRANSFER_DROPOUT = 0.2
FINE_TUNE_EPOCHS = 15
UNFREEZE_AT = 100

#preprocessing
def decode_and_resize(image_path, img_size=IMG_SIZE):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32)
    return image

def load_image_with_label(image_path, label):
    image = decode_and_resize(image_path)
    return image, label

def load_image_only(image_path):
    image = decode_and_resize(image_path)
    return image

def make_paths_from_csv(df, image_root, image_col):
    # If the CSV already stores full paths, this still works because os.path.join
    # will keep the absolute path if image_col is absolute.
    return df[image_col].apply(lambda x: os.path.join(image_root, str(x))).tolist()

def load_datasets_from_csv(
    train_csv,
    image_root,
    img_size,
    batch_size,
    image_col="image",
    label_col="category",
    val_split=0.2,
    seed=42
):
    df = read_csv(train_csv)

    # Shuffle before splitting
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Build label vocabulary from training CSV
    class_names = sorted(df[label_col].astype(str).unique().tolist())
    label_lookup = layers.StringLookup(vocabulary=class_names, num_oov_indices=0)

    # Train/validation split
    val_size = int(len(df) * val_split)
    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()

    train_paths = make_paths_from_csv(train_df, image_root, image_col)
    val_paths   = make_paths_from_csv(val_df, image_root, image_col)

    train_labels = train_df[label_col].astype(str).tolist()
    val_labels   = val_df[label_col].astype(str).tolist()

    train_labels = label_lookup(tf.constant(train_labels))
    val_labels   = label_lookup(tf.constant(val_labels))

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds   = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    train_ds = train_ds.map(load_image_with_label, num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(load_image_with_label, num_parallel_calls=AUTOTUNE)

    return train_ds, val_ds, class_names

def load_test_dataset(test_csv, image_root, image_col="image"):
    df = read_csv(test_csv)
    test_paths = make_paths_from_csv(df, image_root, image_col)
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(load_image_only, num_parallel_calls=AUTOTUNE)
    return test_ds, df

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
    if MODEL_TYPE != "transfer":
        ds = ds.map(lambda x, y: (normalization_layer(x), y),
                    num_parallel_calls=AUTOTUNE)

    if training and MODEL_TYPE != "transfer":
        ds = ds.map(lambda x, y: (data_augmentation(x), y),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
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

#Custom CNN (no pretrained models)
def build_custom_cnn_model(num_classes):
    inputs = layers.Input(shape=IMG_SIZE + (3,))

    x = inputs
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

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

# Transfer
def build_transfer_model(
    num_classes,
    weights=TRANSFER_WEIGHTS,
    dropout=TRANSFER_DROPOUT
):
    backbone_cls = tf.keras.applications.MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = backbone_cls(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=weights
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

#training
def compile_model(model, learning_rate=INITIAL_LR):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Evaluation
def get_labels(model, dataset):
    y_true = []
    y_pred = []

    for batch_images, batch_labels in dataset:
        preds = model.predict(batch_images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        y_true.extend(batch_labels.numpy())
        y_pred.extend(pred_labels)

    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=3),
    keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

train_ds, val_ds, class_names = load_datasets_from_csv(TRAIN_CSV, IMAGE_DIR, IMG_SIZE, BATCH_SIZE)

train_ds = prepare(train_ds, training=True)
val_ds   = prepare(val_ds, training=False)

num_classes = len(class_names)

if   MODEL_TYPE == "cnn": model = build_cnn_model(num_classes)
elif MODEL_TYPE == "vit": model = build_vit_model(num_classes)
elif MODEL_TYPE == "transfer":
    model, base_model = build_transfer_model(num_classes)
else:
    raise ValueError("MODEL_TYPE must be one of: cnn, vit, transfer")

model = compile_model(model)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

if MODEL_TYPE == "transfer" and FINE_TUNE_EPOCHS > 0:
    base_model.trainable = True
    for layer in base_model.layers[:UNFREEZE_AT]:
        layer.trainable = False

    model = compile_model(model, learning_rate=FINE_TUNE_LR)

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=callbacks
    )

#eval
loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc:.4f}")

y_true, y_pred = get_labels(model, val_ds)

macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

plot_confusion_matrix(y_true, y_pred, class_names)

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
