import gc
import torch
from loguru import logger
from transformers import MBartForConditionalGeneration, MBartTokenizer


def load_summary_model(device):
    """
    Загружает модель и токенайзер для суммаризации текста с использованием MBart.

    Parameters
    ----------
    device : torch.device
        Устройство, на которое будет загружена модель (CPU или GPU).

    Returns
    -------
    tokenizer : MBartTokenizer
        Токенайзер для суммаризации текста.
    model : MBartForConditionalGeneration
        Модель для генерации суммаризации текста.
    """
    logger.debug("Downloading summarization model.")
    model_name = "IlyaGusev/mbart_ru_sum_gazeta"
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)  # Загрузка модели на указанное устройство
    return tokenizer, model


def summarize_description(tokenizer, model, description, device):
    """
    Выполняет суммаризацию текста с использованием предобученной модели MBart.

    Обрезает текст, если его длина превышает 1000 символов, и генерирует краткое содержание на основе входного описания.

    Parameters
    ----------
    tokenizer : MBartTokenizer
        Токенайзер для обработки текста перед подачей в модель.
    model : MBartForConditionalGeneration
        Модель MBart для генерации суммаризации.
    description : str
        Входное описание, которое нужно суммировать.
    device : torch.device
        Устройство, на котором выполняется инференс (CPU или GPU).

    Returns
    -------
    summary : str
        Сгенерированная суммаризация текста.
    """
    # Ограничиваем длину описания до 1000 символов
    if len(description) > 1000:
        description = description[:1000] + "..."

    # Преобразуем описание в тензор input_ids
    input_ids = tokenizer(
        [description],
        max_length=600,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    # Генерация суммаризации с помощью модели
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, max_length=150, no_repeat_ngram_size=3, num_beams=4
        )[0]

    # Декодируем сгенерированные ID в текст
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Очистка памяти
    del input_ids, output_ids
    gc.collect()
    torch.cuda.empty_cache()

    return summary
