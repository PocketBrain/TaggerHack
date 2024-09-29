import io

from fastapi import APIRouter, UploadFile
from fastapi.params import Depends, Form

from schemas.video import Video, VideoSchema
from services.video import VideoService

router = APIRouter(prefix="/api/v1/video", tags=["video"])


@router.get(
    "/{uuid}",
    summary="получение видео по id",
    response_model=VideoSchema,
)
async def get(uuid: str, video_service: VideoService = Depends(VideoService)):
    video = await video_service.get(uuid)

    return video


@router.get(
    "/",
    summary="получение списка видео",
    response_model=list[VideoSchema],
)
async def list(video_service: VideoService = Depends(VideoService)):
    videos = await video_service.list()

    return videos


@router.post(
    "/link",
    summary="создание видео из rutube ссылки",
    response_model=VideoSchema,
)
async def create_from_link(
    link: str, video_service: VideoService = Depends(VideoService)
):
    video = await video_service.create_from_rutube_link(link)

    return video


@router.post(
    "/file",
    summary="создание видео из файла",
    response_model=VideoSchema,
)
async def create_from_file(
    file: UploadFile, video_service: VideoService = Depends(VideoService),
    description: str = Form()
):
    file_bytes = await file.read()
    video = await video_service.create_from_file(io.BytesIO(file_bytes), file.filename, description)

    return video
