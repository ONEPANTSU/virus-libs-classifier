import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
from tensorflow import keras

from utils import encode_features, vectorize_features

# Настраиваемые гипер-параметры
DROPOUT = 0.2
DENSE_UNITS = 64
L1 = 1e-5
L2 = 1e-4
EPOCHS = 50
BATCH_SIZE = 128


def plot(history):
    """
    Функция для построения и сохранения графиков обучения
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure()
    plt.plot(loss, "bo", label="Training Loss")
    plt.plot(val_loss, "b", label="Validation Loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("plots/train_val_losses.png")
    plt.show()

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.figure()
    plt.plot(acc, "bo", label="Training Accuracy")
    plt.plot(val_acc, "b", label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracies")
    plt.savefig(
        "plots/train_val_accuracies.png",
    )
    plt.show()


def main():
    # Загрузка тренировочных и валидационных данных
    train_data = pd.read_csv("data/train.tsv", sep="\t")
    train_target = train_data.is_virus.to_numpy().astype("uint8")
    train_features = train_data["libs"].apply(lambda x: x.split(","))

    val_data = pd.read_csv("data/val.tsv", sep="\t")
    val_target = val_data.is_virus.to_numpy().astype("uint8")
    val_features = val_data["libs"].apply(lambda x: x.split(","))

    # Преобразование входных данных в векторы
    word_to_index, max_index = encode_features(train_features)
    train_features_vector = vectorize_features(train_features)
    val_features_vector = vectorize_features(val_features)

    # Создание и обучение модели с двумя полносвязными слоями
    inputs = keras.Input(shape=(max_index + 1,))
    x = layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=keras.regularizers.L1L2(l1=L1, l2=L2),
    )(inputs)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True)]
    history = model.fit(
        train_features_vector,
        train_target,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_features_vector, val_target),
        callbacks=callbacks,
    )

    plot(history)


if __name__ == "__main__":
    main()
