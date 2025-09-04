import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Załadowanie danych CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizacja danych

# Budowa prostego modelu CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Trening modelu
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Funkcja do obliczania mapy saliency
def compute_saliency_map(model, image, class_index):
    image = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)  # Przekształcenie obrazu
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        class_score = predictions[:, class_index]
    
    gradients = tape.gradient(class_score, image)  # Obliczanie gradientów
    saliency_map = tf.abs(gradients[0])  # Wartości bezwzględne gradientów
    return saliency_map

# Wybór obrazu i etykiety z danych testowych
image_index = 0
image = x_test[image_index]
label = y_test[image_index][0]

# Obliczenie mapy saliency
saliency_map = compute_saliency_map(model, image, label)

# Wizualizacja oryginalnego obrazu i mapy saliency
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax1.set_title("Oryginalny obraz")
ax1.axis('off')

ax2.imshow(saliency_map, cmap='hot')
ax2.set_title("Mapa Saliency")
ax2.axis('off')

plt.show()
