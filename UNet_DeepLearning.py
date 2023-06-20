# Emre Karataş


import tensorflow as tf
import random
import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import pickle
import cv2
import time

seed = 42
np.random.seed = seed

IMG_WIDTH = 720
IMG_HEIGHT = 720
IMG_CHANNELS = 3

PATH = 'images/'
model_path = 'models/U_Net/U_Net_Model'
segmentedPath = 'images/Segmented_images_DeepL/segmented_'
U_Net_outputs = 'outputs/U_Net_DeepLearning_Outputs/DeepLearning_Outputs.txt'


X_train = np.zeros((10, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((10, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Load training images and masks')
for subject_id in range(1,11):   
    img = imread(PATH + 'Train_images/' + 'subject_' + str(subject_id) + '.jpg')[:,:,:IMG_CHANNELS]  
    X_train[subject_id-1] = img  #Fill empty X_train with values from img
    img_mask = imread(PATH + 'Train_masks/' + 'subject_' + str(subject_id) + '.png')[:,:,:1]
    #img_mask = np.expand_dims(img_mask, axis=2)    
    Y_train[subject_id-1] = img_mask   
print(X_train.shape)

print('Done!')



#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')]



start_time = time.time()
results = model.fit(X_train[:int(X_train.shape[0]*0.9)], Y_train[:int(X_train.shape[0]*0.9)], validation_split=0.1, batch_size=16, epochs=50, callbacks=callbacks)
end_time = time.time()

####################################


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_test = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1) # last image is used for test !!


# Print accuracy on train and test data
train_acc = results.history['accuracy'][-1]
test_acc = results.history['val_accuracy'][-1]
print("\n\n\nAccuracy on train data: {:.2f}%".format(train_acc * 100))
print("Accuracy on test data: {:.2f}%".format(test_acc * 100))

# Print training time
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))

# Compute and print IoU (Intersection over Union)
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

iou_train = calculate_iou(Y_train[:int(X_train.shape[0]*0.9)], preds_train)
iou_test = calculate_iou(Y_train[int(X_train.shape[0]*0.9):], preds_test)
print("IoU on train data: {:.4f}".format(iou_train))
print("IoU on test data: {:.4f}".format(iou_test))

with open(U_Net_outputs,"w") as U_Net_results:
    U_Net_results.write(f"U-Net Training Accuracy: {train_acc:.4f}"+'\n')
    U_Net_results.write(f"U-Net Test Accuracy: {test_acc:.4f}"+'\n')
    U_Net_results.write(f"U-Net Training Time: {training_time:.4f} seconds"+'\n')
    U_Net_results.write(f"U-Net IoU on train data: {iou_train:.4f}"+'\n')
    U_Net_results.write(f"U-Net IoU on test data: {iou_test:.4f}"+'\n')
U_Net_results.close()



# Perform a sanity check
# Saving segmented train images
for ix in range(0,9):
    plt.imsave(segmentedPath + str(ix+1) + '.jpg', np.squeeze(preds_train[ix]), cmap ='jet')


# Save deep learning model
pickle.dump(model, open(model_path, 'wb'))
loaded_model = pickle.load(open(model_path, 'rb'))
# Predict image using saved model
result = loaded_model.predict(X_train[int(X_train.shape[0]*0.9):])
img_test = cv2.cvtColor(X_train[-1], cv2.COLOR_BGR2GRAY)
segmented = result.reshape((img_test.shape))

plt.subplot(221)
plt.imshow(X_train[-1])
plt.subplot(222)
test_mask = cv2.imread('images/Train_masks/subject_10.png')
test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
plt.imshow(test_mask, cmap ='jet')
plt.subplot(224)
plt.imshow(segmented, cmap ='jet')
# Save test image
plt.imsave(segmentedPath + str(10) + '.jpg', segmented, cmap ='jet')
