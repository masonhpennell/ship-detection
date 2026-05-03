import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_CSV = os.path.join(DATASET_DIR, 'test.csv')
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
MASK_DIR = os.path.join(DATASET_DIR, 'masks')

IMG_SIZE=(224,224)
BATCH_SIZE=8
EPOCHS=40
NUM_CLASSES=6
MODEL_TYPE='unet' # unet, deeplab, fcn
INITIAL_LR=1e-4
AUTOTUNE=tf.data.AUTOTUNE

# ---- Segmentation data loaders (replaces classification labels) ----
def read_image(path):
    x=tf.io.read_file(path)
    x=tf.image.decode_image(x, channels=3, expand_animations=False)
    x=tf.image.resize(x, IMG_SIZE)
    x=tf.cast(x, tf.float32)/255.0
    x.set_shape(IMG_SIZE+(3,))
    return x

def read_mask(path):
    x=tf.io.read_file(path)
    x=tf.image.decode_image(x, channels=1, expand_animations=False)
    x=tf.image.resize(x, IMG_SIZE, method='nearest')
    x=tf.cast(x, tf.int32)
    x=tf.squeeze(x, axis=-1)
    x.set_shape(IMG_SIZE)
    return x

def load_pair(img_path, mask_path):
    return read_image(img_path), read_mask(mask_path)

def make_paths(df):
    imgs=[os.path.join(IMAGE_DIR, f) for f in df['image'].astype(str)]
    masks=[os.path.join(MASK_DIR, os.path.splitext(f)[0]+'.png') for f in df['image'].astype(str)]
    return imgs,masks

def build_ds(csv_path, val_split=0.2):
    df=pd.read_csv(csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
    n=int(len(df)*val_split)
    val_df, train_df=df.iloc[:n], df.iloc[n:]
    tr_i,tr_m=make_paths(train_df); va_i,va_m=make_paths(val_df)
    train=tf.data.Dataset.from_tensor_slices((tr_i,tr_m)).map(load_pair,num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val=tf.data.Dataset.from_tensor_slices((va_i,va_m)).map(load_pair,num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return train,val

# ---- Models ----
def conv_block(x,f):
    x=layers.Conv2D(f,3,padding='same',activation='relu')(x)
    x=layers.Conv2D(f,3,padding='same',activation='relu')(x)
    return x

def build_unet():
    base=tf.keras.applications.EfficientNetB0(include_top=False,weights='imagenet',input_shape=IMG_SIZE+(3,))
    skips=[base.get_layer(n).output for n in ['block2a_expand_activation','block3a_expand_activation','block4a_expand_activation','block6a_expand_activation']]
    x=base.output
    for s,f in zip(reversed(skips),[256,128,64,32]):
        x=layers.UpSampling2D()(x)
        x=layers.Concatenate()([x,s])
        x=conv_block(x,f)
    x=layers.UpSampling2D()(x)
    x=conv_block(x,32)
    out=layers.Conv2D(NUM_CLASSES,1,activation='softmax')(x)
    return keras.Model(base.input,out)

def build_fcn():
    inp=keras.Input(shape=IMG_SIZE+(3,))
    x=conv_block(inp,32); x=layers.MaxPool2D()(x)
    x=conv_block(x,64)
    x=layers.UpSampling2D(size=2)(x)
    out=layers.Conv2D(NUM_CLASSES,1,activation='softmax')(x)
    return keras.Model(inp,out)

def build_deeplab():
    return build_unet()

# ---- Metrics/Losses ----
def dice_coef(y_true,y_pred):
    y_true=tf.one_hot(tf.cast(y_true,tf.int32),NUM_CLASSES)
    y_true=tf.reshape(y_true,[-1,NUM_CLASSES])
    y_pred=tf.reshape(y_pred,[-1,NUM_CLASSES])
    inter=tf.reduce_sum(y_true*y_pred)
    den=tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)
    return (2*inter+1)/(den+1)

def pixel_acc(y_true,y_pred):
    pred=tf.argmax(y_pred,axis=-1,output_type=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true,pred),tf.float32))

miou=tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES,sparse_y_true=True,sparse_y_pred=False)

# ---- Train ----
train_ds,val_ds=build_ds(TRAIN_CSV)
model = build_unet() if MODEL_TYPE=='unet' else build_fcn() if MODEL_TYPE=='fcn' else build_deeplab()
model.compile(optimizer=keras.optimizers.Adam(INITIAL_LR), loss='sparse_categorical_crossentropy', metrics=[pixel_acc,dice_coef,miou])
callbacks=[keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True), keras.callbacks.ReduceLROnPlateau(patience=3), keras.callbacks.ModelCheckpoint('best_segmentation.keras',save_best_only=True)]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ---- Evaluation + visualization (replaces confusion matrix / IG) ----
loss,pa,dice,mi = model.evaluate(val_ds)
print({'loss':loss,'pixel_acc':pa,'dice':dice,'miou':mi})

os.makedirs('predictions',exist_ok=True)
for imgs,masks in val_ds.take(1):
    preds=model.predict(imgs)
    pm=tf.argmax(preds,axis=-1)
    for i in range(min(3,imgs.shape[0])):
        fig,ax=plt.subplots(1,4,figsize=(14,4))
        ax[0].imshow(imgs[i]); ax[0].set_title('image')
        ax[1].imshow(masks[i]); ax[1].set_title('gt')
        ax[2].imshow(pm[i]); ax[2].set_title('pred')
        ax[3].imshow(imgs[i]); ax[3].imshow(pm[i],alpha=0.4); ax[3].set_title('overlay')
        [a.axis('off') for a in ax]
        plt.tight_layout(); plt.savefig(f'predictions/sample_{i}.png'); plt.close()
