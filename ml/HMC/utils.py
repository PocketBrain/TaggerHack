import numpy as np
import torch

from ml.HMC.model import HierarchicalMultimodalClassifier


def load_hmc_model(path, device):
    """
    Загружает и возвращает иерархическую мультимодальную модель классификации вместе с бинаризаторами меток.

    Параметры
    ----------
    path : str
        Путь к сохраненной модели и бинаризаторам меток.
    device : torch.device
        Устройство, на котором будет загружена модель (например, 'cuda' или 'cpu').

    Возвращаемое значение
    ---------------------
    tuple : (HierarchicalMultimodalClassifier, MultiLabelBinarizer, MultiLabelBinarizer, MultiLabelBinarizer)
        Загруженная модель и бинаризаторы меток для трех уровней иерархии.
    """
    # Загрузка сохраненных данных модели и бинаризаторов меток
    saved_data = torch.load(path)
    level1_mlb = saved_data["level1_mlb"]
    level2_mlb = saved_data["level2_mlb"]
    level3_mlb = saved_data["level3_mlb"]

    # Определение размеров выходных слоев для каждого уровня
    level1_size = len(level1_mlb.classes_)
    level2_size = len(level2_mlb.classes_)
    level3_size = len(level3_mlb.classes_)

    # Инициализация модели с правильными размерами выходов
    model = HierarchicalMultimodalClassifier(
        embedding_dim=1024 * 3,
        hidden_dim=512,
        level1_size=level1_size,
        level2_size=level2_size,
        level3_size=level3_size,
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model, level1_mlb, level2_mlb, level3_mlb


def build_parent_mapping(level1_mlb, level2_mlb, level3_mlb, device):
    """
    Создает отображение родительских меток для иерархических уровней меток.

    Параметры
    ----------
    level1_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 1.
    level2_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 2.
    level3_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 3.
    device : torch.device
        Устройство для выполнения расчетов (например, 'cuda' или 'cpu').

    Возвращаемое значение
    ---------------------
    dict
        Словарь, содержащий отображение родительских меток между уровнями в виде тензоров.
    """
    l2_to_l1 = []
    for l2_tag in level2_mlb.classes_:
        l1_tag = l2_tag.split(":")[0]
        if l1_tag in level1_mlb.classes_:
            l1_index = np.where(level1_mlb.classes_ == l1_tag)[0][0]
        else:
            l1_index = -1
        l2_to_l1.append(l1_index)

    l3_to_l2 = []
    for l3_tag in level3_mlb.classes_:
        l2_tag = ":".join(l3_tag.split(":")[:2])
        if l2_tag in level2_mlb.classes_:
            l2_index = np.where(level2_mlb.classes_ == l2_tag)[0][0]
        else:
            l2_index = -1
        l3_to_l2.append(l2_index)

    parent_mapping = {
        "l2_to_l1": torch.tensor(l2_to_l1, dtype=torch.long).to(device),
        "l3_to_l2": torch.tensor(l3_to_l2, dtype=torch.long).to(device),
    }
    return parent_mapping


def decode_predictions(y1_pred, y2_pred, y3_pred, level1_mlb, level2_mlb, level3_mlb):
    """
    Декодирует предсказания моделей в человекочитаемые теги, удаляя избыточные метки на более высоких уровнях.

    Параметры
    ----------
    y1_pred : np.ndarray
        Предсказанные метки для уровня 1.
    y2_pred : np.ndarray
        Предсказанные метки для уровня 2.
    y3_pred : np.ndarray
        Предсказанные метки для уровня 3.
    level1_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 1.
    level2_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 2.
    level3_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 3.

    Возвращаемое значение
    ---------------------
    list of str
        Список строк с декодированными иерархическими метками для каждого примера.
    """
    y1_tags = level1_mlb.inverse_transform(y1_pred)
    y2_tags = level2_mlb.inverse_transform(y2_pred)
    y3_tags = level3_mlb.inverse_transform(y3_pred)

    combined_tags = []
    for l1, l2, l3 in zip(y1_tags, y2_tags, y3_tags):
        final_tags = set()

        for tag in l3:
            final_tags.add(tag)

        l3_parents = set(":".join(tag.split(":")[:-1]) for tag in l3 if ":" in tag)

        for tag in l2:
            if tag not in l3_parents:
                final_tags.add(tag)

        l2_parents = set(tag.split(":")[0] for tag in l2 if ":" in tag)

        for tag in l1:
            if tag not in l2_parents:
                final_tags.add(tag)

        combined_tags.append(", ".join(sorted(final_tags)))

    return combined_tags


def compute_class_weights(encoded_labels):
    """
    Вычисляет веса классов на основе частоты встречаемости меток.

    Параметры
    ----------
    encoded_labels : np.ndarray
        Массив закодированных меток (бинарные значения).

    Возвращаемое значение
    ---------------------
    torch.Tensor
        Веса классов, нормализованные по сумме.
    """
    class_counts = encoded_labels.sum(axis=0)
    total_counts = class_counts.sum()
    class_weights = total_counts / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()
    return torch.tensor(class_weights, dtype=torch.float)
