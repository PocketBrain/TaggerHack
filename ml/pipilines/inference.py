import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from ml.HMC.model import HierarchicalMultimodalClassifier
from ml.HMC.data_loader import VideoDataset
from ml.HMC.utils import build_parent_mapping, decode_predictions, load_hmc_model


def inference(imagebind, whisper, summary_tokenizer, summary_model, hmc_path, device):
    """
    Выполняет процесс инференса для мультимодального иерархического классификатора.

    Параметры
    ----------
    imagebind : объект
        Модель для получения эмбеддингов из изображений и аудио.
    whisper : объект
        Модель для преобразования аудиофайлов в текст.
    summary_tokenizer : объект
        Токенайзер для создания текстовых резюме из описаний.
    summary_model : объект
        Модель для генерации кратких резюме по тексту.
    hmc_path : str
        Путь к сохраненной модели иерархического мультимодального классификатора (HMC).
    device : torch.device
        Устройство для вычислений (например, 'cuda' или 'cpu').

    Описание
    --------
    1. Загружает сохраненную модель иерархического мультимодального классификатора (HMC) и необходимые мультибинарные
       кодировщики для каждого уровня иерархии.
    2. Строит соответствие между уровнями родительских и дочерних тегов.
    3. Загружает тестовый набор данных из CSV-файла и фильтрует его для видео, которые существуют на диске.
    4. Создает тестовый набор данных для инференса с помощью VideoDataset.
    5. Для каждого видео получает эмбеддинги и предсказывает теги на трех уровнях иерархии.
    6. Применяет маски к предсказаниям, чтобы учесть иерархическую структуру меток.
    7. Декодирует предсказания в человекочитаемые теги.
    8. Сохраняет предсказания в CSV-файл.

    Возвращаемое значение
    ---------------------
    None
    """

    model, level1_mlb, level2_mlb, level3_mlb = load_hmc_model(hmc_path, device)

    # Создаем сопоставление родительских меток
    parent_mapping = build_parent_mapping(level1_mlb, level2_mlb, level3_mlb, device)

    # Загружаем тестовый набор данных
    test_df = pd.read_csv("ml/dataset/test.csv")

    # Фильтруем DataFrame для включения только существующих видео
    test_df["video_exists"] = test_df["video_id"].apply(
        lambda x: os.path.exists(f"data/videos/{x}.mp4")
    )
    test_df = test_df[test_df["video_exists"]].reset_index(drop=True)
    test_df = test_df.drop(columns=["video_exists"])

    # Поскольку тестовые данные не содержат теги, создаем пустые столбцы для закодированных меток уровней
    test_df["level1_encoded"] = [[] for _ in range(len(test_df))]
    test_df["level2_encoded"] = [[] for _ in range(len(test_df))]
    test_df["level3_encoded"] = [[] for _ in range(len(test_df))]

    # Создаем тестовый набор данных
    test_dataset = VideoDataset(
        imagebind,
        whisper,
        summary_tokenizer,
        summary_model,
        test_df,
        level1_mlb,
        level2_mlb,
        level3_mlb,
        preprocess=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_predictions = []

    with torch.no_grad():
        for embeddings, _, _, _ in tqdm(test_loader):
            embeddings = embeddings.to(device)

            # Выполняем предсказание тегов для всех уровней
            y1_pred, y2_pred, y3_pred = model(embeddings)

            # Преобразуем предсказания в бинарные метки
            y1_pred_binary = (y1_pred.cpu().numpy() > 0.5).astype(float)
            y2_pred_binary = (y2_pred.cpu().numpy() > 0.5).astype(float)
            y3_pred_binary = (y3_pred.cpu().numpy() > 0.5).astype(float)

            # Применяем иерархические маски
            mask2 = y1_pred_binary[:, parent_mapping["l2_to_l1"].cpu().numpy()]
            y2_pred_binary = y2_pred_binary * mask2

            mask3 = y2_pred_binary[:, parent_mapping["l3_to_l2"].cpu().numpy()]
            y3_pred_binary = y3_pred_binary * mask3

            # Декодируем предсказания в теги
            pred_tags = decode_predictions(
                y1_pred_binary,
                y2_pred_binary,
                y3_pred_binary,
                level1_mlb,
                level2_mlb,
                level3_mlb,
            )
            all_predictions.extend(pred_tags)

    test_df["predicted_tags"] = all_predictions
    save_df = test_df[
        [
            "video_id",
            "predicted_tags",
        ]
    ]
    # Сохраняем предсказания
    save_df.to_csv("ml/dataset/test.csv", index=False)
    print("Predictions saved to 'submit.csv'")


if __name__ == "__main__":
    from ml.lifespan import imagebind, whisper, summary_tokenizer, summary_model, device
    from ml.constants import HMC_PATH

    inference(imagebind, whisper, summary_tokenizer, summary_model, HMC_PATH, device)
