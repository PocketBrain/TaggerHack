import json
import re
import subprocess
import tempfile
from uuid import uuid4

import httpx
from fastapi import Depends
from loguru import logger

from errors.errors import ErrBadRequest, ErrInternalServer, ErrEntityConflict
from schemas.minio import MinioContentType
from schemas.video import VideoInfo, Video
from services.importer import Importer
from services.minio import MinioService
from services.ml import MlService
from utils.httpx_client import get_httpx_client


class RutubeService(Importer):
    def __init__(
        self,
        http_client: httpx.AsyncClient = Depends(get_httpx_client),
        minio: MinioService = Depends(MinioService),
        ml: MlService = Depends(MlService),
    ):
        self._headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/98.0.4758.132 YaBrowser/22.3.1.892 Yowser/2.5 Safari/537.36",
            "accept": "*/*",
        }

        self._download_link = (
            lambda video_id: f"https://rutube.ru/api/play/options/{video_id}/?format=json"
        )

        self._rutube_url_pattern = re.compile(
            r"^https?://(?:www\.)?rutube\.ru/video/[a-zA-Z0-9]{32}/?$"
        )

        self._http_client = http_client

        self._minio = minio

        self._ml = ml

        self._minio_folder = "videos"

    async def download_video(self, url: str) -> Video:
        logger.debug("Rutube - Service - download_video")
        if not self._validate_rutube_download_link(url):
            raise ErrBadRequest("Invalid Rutube video link")

        video_info = await self.get_video_info(url)

        object_path = await self._upload_to_minio(video_info.title, video_info.description, video_info.download_link)

        return Video(title=video_info.title, object_path=object_path)

    async def _download_video_to_folder(self, url: str, path: str, duration_in_ms: int):
        logger.debug("Rutube - Service - download_video")
        if not self._validate_rutube_download_link(url):
            raise ErrBadRequest("Invalid Rutube video link")

        video_info = await self.get_video_info(url)

        if duration_in_ms != -1 and video_info.duration > duration_in_ms:
            raise ErrEntityConflict("video too long")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_info.download_link,
                    "-c",
                    "copy",
                    f"{path}/{video_info.title}.mp4",
                ],
                check=True,
            )
            logger.debug(f"Video downloaded successfully to temporary file {path}")
        except subprocess.CalledProcessError as e:
            raise ErrInternalServer(f"Error downloading video: {e}")

        with open(f"{path}/{video_info.title}.json", "w") as file:
            json.dump(video_info.model_dump(), file)

    async def _upload_to_minio(self, title: str, description: str, download_link: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
            temp_filename = temp_file.name

            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", download_link, "-c", "copy", temp_filename],
                    check=True,
                )
                logger.debug(
                    f"Video downloaded successfully to temporary file {temp_filename}"
                )
            except subprocess.CalledProcessError as e:
                raise ErrInternalServer(f"Error downloading video: {e}")

            self._ml.tagged_video(title, description, temp_filename)

            try:
                uuid = uuid4()

                object_path = f"{self._minio_folder}/{uuid}.mp4"

                self._minio.create_object_from_file(
                    object_path, temp_filename, MinioContentType.MP4
                )
                logger.debug("Video uploaded successfully to MinIO")
            except Exception as e:
                raise ErrInternalServer(f"Error uploading video to MinIO: {e}")

        return object_path

    async def get_video_info(self, url: str) -> VideoInfo:
        video_id = re.search(r"[a-zA-Z0-9]{32}", url).group()

        info_link = self._download_link(video_id)

        try:
            logger.info(f"getting link to download video from Rutube, link {url}")
            response = await self._http_client.get(info_link, headers=self._headers)
            response.raise_for_status()

            video_info = response.json()
        except httpx.HTTPStatusError as e:
            raise ErrInternalServer(f"Failed to download video from Rutube, err {e}")

        logger.debug(f"json = {video_info}")

        download_link = video_info.get("video_balancer", {}).get("m3u8")
        if download_link is None:
            raise ErrInternalServer(
                f"Failed to get download link from Rutube, link {url}"
            )

        title = video_info.get("title", "")
        description = video_info.get("description", "")
        category = video_info.get("category", {})
        tags = video_info.get("tags", [])
        duration = video_info.get("duration", 0)

        return VideoInfo(
            download_link=download_link,
            title=title,
            description=description,
            category=category.get("name", ""),
            duration=duration,
            tags=[tag.get("name", "") for tag in tags],
        )

    def _validate_rutube_download_link(self, url: str) -> bool:
        return bool(self._rutube_url_pattern.match(url))
