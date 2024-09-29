import os
from uuid import uuid4

import torch
import torchaudio
from loguru import logger

from ml.HMC.utils import build_parent_mapping, decode_predictions
from ml.imagebind import ModalityType, data
from ml.lifespan import imagebind, whisper, summary_tokenizer, summary_model, hmc, level1_mlb, level2_mlb, level3_mlb
from ml.summarizer.main import summarize_description


class MlService:
    def __init__(self):
        self.imagebind, self.whisper, self.summary_tokenizer, self.summary_model = imagebind, whisper, summary_tokenizer, summary_model

        self.hmc = hmc

        self.cache_dir = "cache"

        self.parent_mapping = build_parent_mapping(level1_mlb, level2_mlb, level3_mlb, )

    def tagged_video(self, title, description, video_path, device="cuda" if torch.cuda.is_available() else "cpu") -> list[str]:
        # Get title embedding
        title_embedding = self._get_text_embedding(title, device)

        # Summarize description and get its embedding
        summarized_description = summarize_description(self.summary_tokenizer, self.summary_model, description, device)
        description_embedding = self._get_text_embedding(summarized_description, device)

        audio_embedding = self._get_audio_embedding(video_path, device)

        # Combine embeddings from different modalities
        embeddings = torch.cat(
            [title_embedding, description_embedding, audio_embedding], dim=-1
        )  # subtitle_embedding, video_embedding

        embeddings = embeddings.to(device)

        embeddings = embeddings.to(device)

        y1_pred, y2_pred, y3_pred = self.hmc(embeddings)

        y1_pred_binary = (y1_pred.cpu().numpy() > 0.5).astype(float)
        y2_pred_binary = (y2_pred.cpu().numpy() > 0.5).astype(float)
        y3_pred_binary = (y3_pred.cpu().numpy() > 0.5).astype(float)

        # Apply hierarchical masks
        mask2 = y1_pred_binary[:, self.parent_mapping["l2_to_l1"].cpu().numpy()]
        y2_pred_binary = y2_pred_binary * mask2

        mask3 = y2_pred_binary[:, self.parent_mapping["l3_to_l2"].cpu().numpy()]
        y3_pred_binary = y3_pred_binary * mask3

        all_predictions = []

        # Decode predictions
        pred_tags = decode_predictions(
            y1_pred_binary,
            y2_pred_binary,
            y3_pred_binary,
            level1_mlb,
            level2_mlb,
            level3_mlb,
        )

        all_predictions.extend(pred_tags)

        return all_predictions

    def _get_text_embedding(self, text, device):
        if not text.strip():
            logger.debug("Empty text input. Returning zero vector.")
            return torch.zeros(1024, device=device)
        inputs = {ModalityType.TEXT: data.load_and_transform_text([text], device)}
        with torch.no_grad():
            embeddings = self.imagebind(inputs)
        text_embedding = embeddings[ModalityType.TEXT][0]
        logger.debug("Computed text embedding.")
        return text_embedding

    def _get_audio_embedding(self, video_path, device):
        if not os.path.exists(video_path):
            logger.warning(
                f"Video file not found for video_path {video_path}. Cannot extract audio."
            )
            return torch.zeros(1024, device=device)
        try:
            logger.debug(f"Loading audio for {video_path}.")
            waveform, sr = torchaudio.load(video_path)

            if sr != 16000:
                logger.debug(
                    f"Resampling audio from {sr} Hz to 16000 Hz for video_path {video_path}."
                )
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # Save temp audio file
            audio_path = f"{self.cache_dir}/temp_audio_{uuid4()}.wav"
            torchaudio.save(audio_path, waveform, 16000)

            inputs = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    [audio_path], device
                )
            }
            with torch.no_grad():
                embeddings = self.imagebind(inputs)
            audio_embedding = embeddings[ModalityType.AUDIO].mean(dim=0)

            os.remove(audio_path)
            logger.debug(f"Computed audio embedding for video_path {video_path}.")
        except Exception as e:
            logger.error(f"Error processing audio for video_path {video_path}: {e}")
            audio_embedding = torch.zeros(1024, device=device)
        return audio_embedding