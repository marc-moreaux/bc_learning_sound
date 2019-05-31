import keras
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, GlobalMaxPooling1D, Activation
import dataset


n_classes = 10

inputs = Input(tensor=inputs)
x = inputs

for _ in range(12):
    x = Conv1D(64, 3, strides=2, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)

x = Conv1D(n_classes, 3, strides=2)(x)
x = GlobalMaxPooling1D()(x)
y = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=y)


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              target_tensors=[target])

model.summary()
model.fit(epochs=1, steps_per_epoch=64)  # starts training



