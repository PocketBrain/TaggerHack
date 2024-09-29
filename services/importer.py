from abc import ABC, abstractmethod

from models.video import Video
from schemas.video import VideoInfo


class Importer(ABC):
    @abstractmethod
    def download_video(self, url: str) -> Video: ...

    @abstractmethod
    def get_video_info(self, url: str) -> VideoInfo: ...
