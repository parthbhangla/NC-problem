import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNetV2

# Output directory
OUTPUT_DIR = "output_results_mobilenet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data Generator
class DataGenerator(Sequence):
    def __init__(self, path='./neuralzome_crate_local/2024-01-31-09-51-48/rgb/', 
                 batch_size=16, shuffle=True, target_size=(224, 224)):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size

        self.images = []
        for file in sorted(os.listdir(self.path)):
            if file.endswith('.jpg'):
                yaml_path = os.path.join(self.path, file.replace('.jpg', '.yaml'))
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if data.get('crates'):
                            self.images.append(file)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    def __getitem__(self, index):
        batch_files = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        keypoints = []

        for file in batch_files:
            img_path = os.path.join(self.path, file)
            yaml_path = os.path.join(self.path, file.replace('.jpg', '.yaml'))

            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            img = img.resize((self.target_size[1], self.target_size[0]))
            img_array = np.array(img).astype('float32') / 255.0

            with open(yaml_path, 'r') as f:
                crate = yaml.safe_load(f)['crates'][0]

            x_scale = self.target_size[1] / original_size[0]
            y_scale = self.target_size[0] / original_size[1]

            kp = np.array([
                crate['x0'] * x_scale, crate['y0'] * y_scale,
                crate['x1'] * x_scale, crate['y1'] * y_scale,
                crate['x2'] * x_scale, crate['y2'] * y_scale,
                crate['x3'] * x_scale, crate['y3'] * y_scale
            ], dtype='float32')

            images.append(img_array)
            keypoints.append(kp)

        return np.stack(images), np.stack(keypoints)

# Loss function
def keypoint_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Model definition
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

model_more_layer = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.Dense(8)
])

model_more_layer.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=keypoint_loss,
    metrics=['mae']
)

dataset = DataGenerator(target_size=(224, 224))

history_more_layer = model_more_layer.fit(dataset, epochs=40)

model_more_layer.save(os.path.join(OUTPUT_DIR, 'mobilenet_model.keras'))

def visualize_predictions(model, generator, num_images=5):
    for i in range(num_images):
        images, keypoints_true = generator[i]
        preds = model.predict(images)

        for j in range(min(len(images), num_images)):
            img = images[j]
            true_kp = keypoints_true[j].reshape(-1, 2)
            pred_kp = preds[j].reshape(-1, 2)

            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.scatter(true_kp[:, 0], true_kp[:, 1], c='green', label='Actual', s=40)
            plt.scatter(pred_kp[:, 0], pred_kp[:, 1], c='red', marker='x', label='Prediction', s=40)
            plt.legend()
            plt.title(f"Image {i}_{j}: Predicted vs Actual")
            plt.axis('off')

            plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_{i}_{j}.png'))
            plt.close()

visualize_predictions(model_more_layer, dataset, num_images=5)

plt.figure()
plt.plot(history_more_layer.history['loss'], label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
plt.close()

plt.figure()
plt.plot(history_more_layer.history['mae'], label='MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_mae.png'))
plt.close()
