import torch
import gc
from torch.utils.data import Dataset
import torchaudio
from PIL import Image
import os
import cv2
import numpy as np
from ml.imagebind.models.imagebind_model import ModalityType
from ml.imagebind import data
from tqdm import tqdm
from torchvision import transforms
from loguru import logger
from moviepy.editor import VideoFileClip

from ml.summarizer.main import summarize_description


def load_and_transform_video_data(video_path, device, chunk_size=32):
    """
    Загружает и преобразует данные видео по частям для обработки на устройстве.

    Параметры
    ----------
    video_path : str
        Путь к видеофайлу.
    device : torch.device
        Устройство для вычислений ('cuda' или 'cpu').
    chunk_size : int, по умолчанию 32
        Размер блока кадров, обрабатываемого за одну итерацию.

    Описание
    --------
    Эта функция извлекает кадры из видео, преобразует их в оттенки серого, изменяет размер до 224x224 и нормализует.
    Кадры обрабатываются частями (по `chunk_size`), чтобы избежать переполнения памяти на GPU. Преобразованные кадры
    отправляются на заданное устройство для последующей обработки.

    Возвращаемое значение
    ---------------------
    Generator из torch.Tensor
        Возвращает генератор тензоров обработанных кадров, готовых для подачи в модель.
    """
    video_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Преобразование в оттенки серого
            transforms.Lambda(lambda img: img.repeat(3, 1, 1)),  # Повторение канала для 3-канального тензора
            transforms.Resize((224, 224)),  # Изменение размера для модели
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    all_video = []
    with VideoFileClip(video_path) as video:
        for frame in video.iter_frames():
            frame_tensor = torch.tensor(frame).float().permute(2, 0, 1) / 255.0
            transformed_frame = video_transform(frame_tensor).unsqueeze(0)
            all_video.append(transformed_frame)

            if len(all_video) == chunk_size:
                yield torch.cat(all_video, dim=0).to(device)
                all_video = []
                torch.cuda.empty_cache()
                gc.collect()

        if all_video:
            yield torch.cat(all_video, dim=0).to(device)
            torch.cuda.empty_cache()
            gc.collect()


class VideoDataset(Dataset):
    """
    Класс датасета для обработки видеофайлов и извлечения эмбеддингов мультимодальных данных.

    Параметры
    ----------
    imagebind : объект
        Модель для извлечения эмбеддингов из видео, аудио и текста.
    whisper : объект
        Модель для преобразования аудио в текст.
    summary_tokenizer : объект
        Токенайзер для генерации текстовых резюме.
    summary_model : объект
        Модель для генерации кратких текстовых описаний.
    df : pandas.DataFrame
        DataFrame с метаданными и информацией о видео.
    level1_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 1.
    level2_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 2.
    level3_mlb : MultiLabelBinarizer
        Бинаризатор меток для уровня 3.
    cache_dir : str, по умолчанию 'cache/'
        Каталог для кэширования предварительно вычисленных эмбеддингов.
    preprocess : bool, по умолчанию False
        Если True, эмбеддинги будут вычислены и сохранены заранее.
    device : torch.device, по умолчанию 'cuda' если доступно, иначе 'cpu'
        Устройство для вычислений.

    Описание
    --------
    Этот класс предоставляет функциональность для обработки видео, аудио и текстовых данных с использованием мультимодальных моделей.
    Он также поддерживает кэширование эмбеддингов для ускорения последующих инференсов. Если указан флаг `preprocess`,
    эмбеддинги будут предварительно рассчитаны и сохранены в кэше.
    """

    def __init__(
        self,
        imagebind,
        whisper,
        summary_tokenizer,
        summary_model,
        df,
        level1_mlb,
        level2_mlb,
        level3_mlb,
        cache_dir="cache/",
        preprocess=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.df = df.reset_index(drop=True)
        self.level1_mlb = level1_mlb
        self.level2_mlb = level2_mlb
        self.level3_mlb = level3_mlb
        self.cache_dir = cache_dir
        self.preprocess = preprocess
        self.device = device

        os.makedirs(self.cache_dir, exist_ok=True)

        self.imagebind_model = imagebind
        logger.debug("Initialized ImageBind model.")

        self.whisper_model = whisper
        logger.debug("Initialized Whisper model.")

        self.summary_tokenizer, self.summary_model = summary_tokenizer, summary_model
        logger.debug("Initialized summarization model.")

        if self.preprocess:
            self.precompute_embeddings()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Возвращает эмбеддинги и закодированные метки для указанного видео.

        Параметры
        ----------
        idx : int
            Индекс видео в DataFrame.

        Возвращаемое значение
        ---------------------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            Эмбеддинги и закодированные метки для уровней 1, 2 и 3.
        """
        row = self.df.loc[idx]

        video_id = row["video_id"]
        embeddings_path = os.path.join(self.cache_dir, f"{video_id}_embeddings.pt")

        if not os.path.exists(embeddings_path):
            logger.warning(
                f"Embeddings not found for video_id {video_id}. Precomputing embeddings."
            )
            self.precompute_embedding_for_video(row)

        embeddings = torch.load(embeddings_path).to(self.device)

        y1 = torch.tensor(row["level1_encoded"], dtype=torch.float32)
        y2 = torch.tensor(row["level2_encoded"], dtype=torch.float32)
        y3 = torch.tensor(row["level3_encoded"], dtype=torch.float32)

        return embeddings, y1, y2, y3

    def precompute_embeddings(self):
        """
        Предварительно вычисляет и сохраняет эмбеддинги для всех видео в датасете.

        Описание
        --------
        Эта функция вычисляет эмбеддинги для всех видео в датасете и сохраняет их в указанной директории кэша.
        """
        logger.info("Starting precomputation of embeddings.")
        for idx in tqdm(range(len(self.df))):
            row = self.df.loc[idx]
            self.precompute_embedding_for_video(row)
        logger.info("Finished precomputing embeddings.")

    def precompute_embedding_for_video(self, row):
        """
        Вычисляет эмбеддинги для одного видео и сохраняет их в файл.

        Параметры
        ----------
        row : pandas.Series
            Строка из DataFrame, содержащая информацию о видео.

        Описание
        --------
        Эта функция получает эмбеддинги текста, описания, аудио и видео (если они доступны) и сохраняет их в файл для указанного видео.
        """
        video_id = row["video_id"]
        title = row["title"]
        description = row["description"]

        embeddings_path = os.path.join(self.cache_dir, f"{video_id}_embeddings.pt")
        if os.path.exists(embeddings_path):
            logger.debug(f"Embeddings already exist for video_id {video_id}. Skipping.")
            return

        logger.debug(f"Processing video_id {video_id}.")

        title_embedding = self.get_text_embedding(title)

        summarized_description = summarize_description(self.summary_tokenizer, self.summary_model, description, self.device)
        description_embedding = self.get_text_embedding(summarized_description)

        audio_embedding = self.get_audio_embedding(video_id)

        embeddings = torch.cat([title_embedding, description_embedding, audio_embedding], dim=-1)
        embeddings = embeddings.cpu()

        torch.save(embeddings, embeddings_path)
        logger.debug(f"Saved embeddings for video_id {video_id}.")

    def get_text_embedding(self, text):
        """
        Получает эмбеддинг для текстового ввода.

        Параметры
        ----------
        text : str
            Текст для получения эмбеддинга.

        Возвращаемое значение
        ---------------------
        torch.Tensor
            Эмбеддинг текста.
        """
        if not text.strip():
            logger.debug("Empty text input. Returning zero vector.")
            return torch.zeros(1024, device=self.device)
        inputs = {ModalityType.TEXT: data.load_and_transform_text([text], self.device)}
        with torch.no_grad():
            embeddings = self.imagebind_model(inputs)
        text_embedding = embeddings[ModalityType.TEXT][0]
        logger.debug("Computed text embedding.")
        return text_embedding

    def get_subtitles(self, video_id):
        """
        Извлекает субтитры из видео с помощью модели Whisper.

        Параметры
        ----------
        video_id : str
            Идентификатор видео для извлечения субтитров.

        Возвращаемое значение
        ---------------------
        str
            Расшифрованный текст субтитров.
        """
        video_path = f"data/videos/{video_id}.mp4"
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found for video_id {video_id}. Cannot extract subtitles.")
            return ""
        try:
            logger.debug(f"Transcribing subtitles for video_id {video_id}.")
            result = self.whisper_model.transcribe(video_path)
            subtitle_text = result["text"]
            logger.debug(f"Transcription completed for video_id {video_id}.")
        except Exception as e:
            logger.error(f"Error transcribing video_id {video_id}: {e}")
            subtitle_text = ""
        return subtitle_text

    def get_audio_embedding(self, video_id):
        """
        Извлекает аудио из видео и вычисляет его эмбеддинг.

        Параметры
        ----------
        video_id : str
            Идентификатор видео для извлечения аудио.

        Возвращаемое значение
        ---------------------
        torch.Tensor
            Эмбеддинг аудио.
        """
        video_path = f"data/videos/{video_id}.mp4"
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found for video_id {video_id}. Cannot extract audio.")
            return torch.zeros(1024, device=self.device)
        try:
            logger.debug(f"Loading audio for video_id {video_id}.")
            waveform, sr = torchaudio.load(video_path)

            if sr != 16000:
                logger.debug(f"Resampling audio from {sr} Hz to 16000 Hz for video_id {video_id}.")
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            audio_path = f"{self.cache_dir}/temp_audio_{video_id}.wav"
            torchaudio.save(audio_path, waveform, 16000)

            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], self.device)}
            with torch.no_grad():
                embeddings = self.imagebind_model(inputs)
            audio_embedding = embeddings[ModalityType.AUDIO].mean(dim=0)

            os.remove(audio_path)
            logger.debug(f"Computed audio embedding for video_id {video_id}.")
        except Exception as e:
            logger.error(f"Error processing audio for video_id {video_id}: {e}")
            audio_embedding = torch.zeros(1024, device=self.device)
        return audio_embedding

    def get_video_embedding(self, video_id):
        """
        Извлекает кадры из видео и вычисляет их эмбеддинги.

        Параметры
        ----------
        video_id : str
            Идентификатор видео для извлечения эмбеддингов.

        Возвращаемое значение
        ---------------------
        torch.Tensor
            Эмбеддинг видео.
        """
        video_path = f"data/videos/{video_id}.mp4"
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found for video_id {video_id}. Cannot extract video frames.")
            return torch.zeros(1024, device=self.device)

        try:
            embeddings_list = []
            with torch.no_grad():
                for video_chunk in load_and_transform_video_data(video_path, self.device):
                    chunk_embeddings = self.imagebind_model({ModalityType.VISION: video_chunk})
                    embeddings_list.append(chunk_embeddings[ModalityType.VISION].mean(dim=0))

                    torch.cuda.empty_cache()
                    gc.collect()

                video_embedding = torch.stack(embeddings_list).mean(dim=0)
                logger.debug(f"Computed video embedding for video_id {video_id}.")
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing video for video_id {video_id}: {e}")
            video_embedding = torch.zeros(1024, device=self.device)

        return video_embedding

    def extract_frames(self, video_path, num_frames=8):
        """
        Извлекает указанное количество кадров из видео.

        Параметры
        ----------
        video_path : str
            Путь к видеофайлу.
        num_frames : int, по умолчанию 8
            Количество кадров для извлечения.

        Возвращаемое значение
        ---------------------
        list of PIL.Image
            Список извлеченных кадров.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            logger.warning(f"Video at {video_path} has zero frames.")
            return []

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                frames.append(pil_image)
            else:
                logger.warning(f"Failed to read frame at index {idx} from video {video_path}.")
        cap.release()
        logger.debug(f"Extracted {len(frames)} frames from video {video_path}.")
        return frames
