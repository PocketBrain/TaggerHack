import uuid

from pydantic import BaseModel

class VideoSchema(BaseModel):
    id: uuid.UUID
    link: str
    title: str
    description: str

class VideoInfo(BaseModel):
    title: str
    download_link: str
    description: str
    category: str
    duration: int
    tags: list[str]


class Video(BaseModel):
    title: str
    object_path: str
