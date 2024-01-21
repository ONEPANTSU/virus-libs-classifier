import pandas as pd
from tensorflow import keras

from utils import vectorize_features


def main():
    # Загрузка тестовых данных и преобразование их в векторную форму
    test_data = pd.read_csv("data/test.tsv", sep="\t")
    test_features = test_data["libs"].apply(lambda x: x.split(","))
    test_features_vector = vectorize_features(test_features)

    # Загрузка обученной модели
    model = keras.models.load_model("model.keras")

    # Создание предсказаний и запись их в test.txt
    test_pred = model.predict(test_features_vector)
    test_pred_binary = (test_pred > 0.5).astype(int).flatten()
    with open("test.txt", "w") as f:
        for pred in test_pred_binary:
            f.write(str(pred) + "\n")


if __name__ == "__main__":
    main()
