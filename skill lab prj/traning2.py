import numpy as np
import tensorflow as tf
import cv2 as cv
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import os

def create_siamese_model(input_shape):
    input_img = layers.Input(shape=input_shape)
    base_network = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    encoded_img = base_network(input_img)
    return models.Model(input_img, encoded_img)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    
    # Cast y_true to the same data type as square_pred and margin_square
    y_true = tf.cast(y_true, square_pred.dtype)
    
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def generate_pairs(X, y, num_pairs=10000):
    pairs = []
    labels = []
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(len(X), size=2, replace=False)
        pairs.append([X[idx1], X[idx2]])
        labels.append(int(y[idx1] == y[idx2]))
    return np.array(pairs), np.array(labels)

def train_siamese_model(X, y, input_shape, num_epochs=10, batch_size=32):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_siamese_model(input_shape)
    model.compile(loss=contrastive_loss, optimizer=optimizers.Adam(), metrics=['accuracy'])
    
    # Reshape input pairs into a single batch of images
    X_train = X_train.reshape(-1, *input_shape)
    X_val = X_val.reshape(-1, *input_shape)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size)
    
    return model

    
    # model = create_siamese_model(input_shape)
    # model.compile(loss=contrastive_loss, optimizer=optimizers.Adam(), metrics=['accuracy'])
    
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size)
    
    # return model

# Example usage
if __name__ == "__main__":
    def loadImages(folderPath):
     images = []
     labels = []
     labelDict = {}

     currentId = 0

     for root, dirs, files in os.walk(folderPath):
         for file in files:
             if file.endswith(".jpeg" or ".jpg"):
                 label = os.path.basename(root)
                 if label not in labelDict:
                     labelDict[label] = currentId
                     currentId += 1

                 imgPath = os.path.join(root, file)
                 img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
                 img = cv.resize(img, (100, 100))
                 images.append(img)
                 labels.append(labelDict[label])  # Move this line inside the loop

     return np.array(images), np.array(labels), labelDict


    folderPath = 'Images'
    X, y, labelDict = loadImages(folderPath)

    X = X.astype('float32') /255.0
    X = X.reshape(-1, 100, 100, 1)

    input_shape = (100, 100, 1)  # Example input shape (replace with your actual input shape)
    num_epochs = 10
    batch_size = 32
    
    # Train the Siamese model
    siamese_model = train_siamese_model(X, y, input_shape, num_epochs, batch_size)
    
    # Save the trained model
    siamese_model.save('siamese_face_model.h5')
