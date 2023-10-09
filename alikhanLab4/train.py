import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

df = pd.read_csv('dataset/train.csv')
df['image_id'] = df['image_id'].apply(lambda x: x + '.jpg')


def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)
    img = img / 255.0  # Масштабирование значений пикселей к диапазону [0, 1]
    return img


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    df,
    directory='dataset/images',  # Путь к директории с изображениями
    x_col='image_id',  # Столбец с именами файлов изображений
    y_col=['healthy', 'multiple_diseases', 'rust', 'scab'],  # Столбцы с метками
    target_size=(150, 150),
    batch_size=32,
    class_mode='raw',  # Для многоклассовой классификации
)


# Создаем модель Sequential
model = Sequential()

# Первый сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

# Второй сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Третий сверточный слой
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Разворачиваем данные перед полносвязным слоем
model.add(Flatten())

# Полносвязные слои
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))  # 4 класса для болезней

# Компилируем модель
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Выводим информацию о модели
model.summary()


print(len(train_generator))
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
)

model.save("model")
