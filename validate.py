import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import colormaps
from tensorflow import keras

from utils import vectorize_features


def plot(con_mat):
    """
    Функция для построения и сохранения матрицы ошибок
    """
    con_mat_norm = np.around(
        con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2
    )
    con_mat_df = pd.DataFrame(
        con_mat_norm,
        index=["Ordinary File", "Virus"],
        columns=["Ordinary File", "Virus"],
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=colormaps["Blues"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("plots/confusion_matrix.png")
    plt.show()


def save_results(con_mat):
    """
    Функция для сохранения результатов в validation.txt
    """
    true_negative = con_mat[0, 0]
    false_negative = con_mat[1, 0]
    false_positive = con_mat[0, 1]
    true_positive = con_mat[1, 1]
    accuracy = (true_positive + true_negative) / sum(con_mat.flatten())
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) != 0
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) != 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )
    output_text = (
        f"True positive: {true_positive}\n"
        f"False positive: {false_positive}\n"
        f"False negative: {false_negative}\n"
        f"True negative: {true_negative}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1: {f1:.4f}\n"
    )
    with open("validation.txt", "w") as f:
        f.write(output_text)

    print(output_text)


def main():
    # Загрузка валидационных данных и преобразование их в векторную форму
    val_data = pd.read_csv("data/val.tsv", sep="\t")
    val_features = val_data["libs"].apply(lambda x: x.split(","))
    val_features_vector = vectorize_features(val_features)

    # Загрузка обученной модели и создание предсказаний
    model = keras.models.load_model("model.keras")
    val_pred = model.predict(val_features_vector)

    val_pred_binary = (val_pred > 0.5).astype(int)
    val_true = val_data.is_virus.values

    # Создание матрицы ошибок
    con_mat = tf.math.confusion_matrix(
        labels=val_true, predictions=val_pred_binary
    ).numpy()
    save_results(con_mat)
    plot(con_mat)


if __name__ == "__main__":
    main()
