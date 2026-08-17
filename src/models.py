"""
Model architectures used in this project:

1. A small custom CNN trained from scratch.
2. VGG16 with ImageNet weights, fine-tuned on the last 10 layers.
3. ResNet101 with ImageNet weights, fully fine-tuned.

Each `create_*` function returns a compiled `tf.keras.Model`.
"""

from tensorflow.keras.applications import VGG16, ResNet101
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from src import config


def create_cnn_model(input_shape: tuple = config.IMAGE_SHAPE, num_classes: int = len(config.CLASS_NAMES)) -> Sequential:
    """A simple 4-block CNN trained from scratch."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_vgg16_model(
    input_shape: tuple = config.IMAGE_SHAPE,
    num_classes: int = len(config.CLASS_NAMES),
    fine_tune_last_n: int = 10,
    learning_rate: float = 1e-4,
) -> Model:
    """VGG16 backbone (ImageNet weights) with the last `fine_tune_last_n` layers unfrozen."""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in base_model.layers[:-fine_tune_last_n]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_last_n:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_resnet101_model(
    input_shape: tuple = config.IMAGE_SHAPE,
    num_classes: int = len(config.CLASS_NAMES),
    learning_rate: float = 1e-4,
) -> Model:
    """ResNet101 backbone (ImageNet weights), fully fine-tuned."""
    base_model = ResNet101(weights="imagenet", include_top=False, input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


MODEL_REGISTRY = {
    "cnn": create_cnn_model,
    "vgg16": create_vgg16_model,
    "resnet101": create_resnet101_model,
}
