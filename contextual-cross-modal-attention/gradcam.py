import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def compute_grad_cam(model, img_array, target_class_index):
    # Convert image to a tensor
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Watch the input image tensor
        tape.watch(img_tensor)
        
        # Forward pass
        predictions = model(img_tensor)
        predicted_class = predictions[:, target_class_index]
    
    # Get the gradient of the predicted class with respect to the feature maps
    grads = tape.gradient(predicted_class, model.layers[-1].output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Get feature maps and compute weighted combination
    last_conv_layer_output = model.layers[-3].output
    heatmap = tf.reduce_mean(last_conv_layer_output * pooled_grads[..., tf.newaxis], axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize heatmap
    heatmap /= tf.reduce_max(heatmap)
    
    return heatmap.numpy()

def visualize_grad_cam(img_array, heatmap):
    plt.imshow(img_array)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.show()

# Load pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Load and preprocess image
img_path = '/home/UNT/vk0318/Documents/Work/Code/MultimodalDeepfakeDetection-master/processed/Train/Fake/fake_0.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = preprocess_input(img_array)

# Predict class
preds = model.predict(img_array[np.newaxis, ...])
predicted_class_index = np.argmax(preds[0])

# Compute Grad-CAM heatmap
heatmap = compute_grad_cam(model, img_array, predicted_class_index)

# Visualize heatmap
visualize_grad_cam(img_array, heatmap)
