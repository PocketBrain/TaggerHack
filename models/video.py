import uuid

from sqlalchemy.orm import Mapped, mapped_column

from models.BaseModel import EntityMeta


class Video(EntityMeta):
    __tablename__ = "video"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    object_path: Mapped[str]
    title: Mapped[str]
    description: Mapped[str]
