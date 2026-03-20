import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model

NUM_CLASSES = 7

def build_relational_cnn():

    inp = Input(shape=(64,64,1))

    # CNN feature extractor
    x = Conv2D(32,(3,3),activation='relu')(inp)
    x = MaxPooling2D()(x)

    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128,(3,3),activation='relu')(x)

    # Relational block (simple)
    f1 = GlobalAveragePooling2D()(x)
    f2 = tf.reduce_mean(x, axis=[1,2])

    r = Concatenate()([f1,f2])

    # Classifier
    r = Dense(128,activation='relu')(r)
    r = Dropout(0.5)(r)
    out = Dense(NUM_CLASSES,activation='softmax')(r)

    model = Model(inp,out)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_relational_cnn()
    model.summary()
