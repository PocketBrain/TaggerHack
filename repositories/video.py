from fastapi import Depends
from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from configs.Database import get_db_connection
from errors.errors import ErrEntityNotFound
from models.video import Video


class VideoRepository:
    def __init__(self, db: AsyncSession = Depends(get_db_connection)):
        self._db = db

    async def get(self, uuid: str) -> Video:
        logger.debug("Video - Repository - get")

        query = select(Video).where(Video.id == uuid)

        result = await self._db.execute(query)

        try:
            video = result.scalar_one()
        except NoResultFound:
            raise ErrEntityNotFound("entity not found")

        return video

    async def create(self, video: Video) -> Video:
        logger.debug("Video - Repository - create")

        self._db.add(video)
        await self._db.commit()
        await self._db.refresh(video)
        return video

    async def list(self) -> list[Video]:
        logger.debug("Video - Repository - list")

        query = select(Video)

        result = await self._db.execute(query)

        return list(result.scalars().all())
