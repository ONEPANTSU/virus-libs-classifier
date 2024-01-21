import pickle

import numpy as np


def encode_features(features):
    """
    Функция для кодирования признаков и сохранение их в файлы
    """
    known_files = set()
    for row in features:
        for file in row:
            known_files.add(file)
    word_to_index = dict()
    max_index = -1
    for index, word in enumerate(known_files):
        word_to_index[word] = index
        max_index = index
    save_to_file(word_to_index, "encoding/word_to_index")
    save_to_file(max_index, "encoding/max_index")
    return word_to_index, max_index


def save_to_file(values, filename):
    """
    Функция для сериализации и сохранения данных в файл
    """
    with open(f"{filename}.pkl", "wb") as file:
        pickle.dump(values, file)


def load_from_file(filename):
    """
    Функция для загрузки сериализованных данных и восстановление объекта
    """
    with open(f"{filename}.pkl", "rb") as file:
        return pickle.load(file)


def vectorize_features(features):
    """
    Преобразование признаков в векторную форму
    """
    try:
        word_to_index = load_from_file("encoding/word_to_index")
        max_index = load_from_file("encoding/max_index")
    except FileNotFoundError:
        word_to_index, max_index = encode_features(features)

    for i in range(len(features)):
        encoded = np.zeros(max_index + 1)
        for file in features[i]:
            if word_to_index.get(file, None):
                encoded[word_to_index[file]] = 1
        features[i] = encoded[:]
    return np.array(features.to_list()).astype("uint8")
