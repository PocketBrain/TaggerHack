import io
from uuid import uuid4

from fastapi.params import Depends
from loguru import logger

from models.video import Video
from repositories.video import VideoRepository
from schemas.minio import MinioContentType
from schemas.video import VideoSchema
from services.minio import MinioService
from services.ml import MlService
from services.rutube import RutubeService


class VideoService:
    def __init__(
        self,
        repo: VideoRepository = Depends(VideoRepository),
        rutube: RutubeService = Depends(RutubeService),
        minio: MinioService = Depends(MinioService),
        ml: MlService = Depends(MlService),
    ):
        self._rutube = rutube
        self._repo = repo
        self._minio = minio
        self._ml = ml

        self._minio_folder = "videos"

    async def create_from_rutube_link(self, url: str) -> VideoSchema:
        logger.debug("Video - Service - create_from_rutube_link")
        rutube_video = await self._rutube.download_video(url)

        video = await self._repo.create(
            Video(
                id=uuid4(),
                object_path=rutube_video.object_path,
                title=rutube_video.title,
                description="",
            )
        )

        return await self._repository_model_to_service(video)

    async def create_from_file(self, file: io.BytesIO, title: str, description: str) -> VideoSchema:
        logger.debug("Video - Service - create_from_file")
        uuid = uuid4()

        object_path = f"{self._minio_folder}/{uuid}.mp4"

        self._minio.create_object_from_byte(
            object_path=object_path,
            file=file,
            content_type=MinioContentType.MP4,
        )

        video = await self._repo.create(
            Video(
                id=uuid,
                object_path=object_path,
                title=title,
                description=description,
            )
        )

        return await self._repository_model_to_service(video)

    async def get(self, uuid: str) -> VideoSchema:
        logger.debug("Video - Service - get")
        video = await self._repo.get(uuid)

        return await self._repository_model_to_service(video)

    async def list(self) -> list[VideoSchema]:
        logger.debug("Video - Service - list")
        videos = await self._repo.list()

        return [await self._repository_model_to_service(video) for video in videos]

    async def _repository_model_to_service(self, req: Video) -> VideoSchema:
        return VideoSchema(
            id=req.id,
            link=self._minio.get_link(object_path=req.object_path),
            title=req.title,
            description=req.description,
        )
