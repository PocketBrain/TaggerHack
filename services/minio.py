import io

from fastapi import Depends
from loguru import logger

from configs import Minio
from configs.Minio import get_minio_client, base_bucket
from schemas.minio import MinioContentType


class MinioService:
    def __init__(self, client: Minio = Depends(get_minio_client)):
        self._client = client
        self.create_bucket(base_bucket)

    def create_object_from_byte(
        self,
        object_path: str,
        file: io.BytesIO,
        content_type: MinioContentType,
        bucket_name: str = base_bucket,
    ) -> str:
        logger.debug("Minio - Service - create_object_from_byte")
        self._client.put_object(
            bucket_name,
            object_path,
            data=file,
            length=file.getbuffer().nbytes,
            content_type=content_type.value,
        )

        return object_path

    def create_object_from_file(
        self,
        object_path: str,
        file: str,
        content_type: MinioContentType,
        bucket_name: str = base_bucket,
    ) -> str:
        logger.debug("Minio - Service - create_object_from_file")
        self._client.fput_object(
            bucket_name,
            object_path,
            file,
            content_type=content_type.value,
        )

        return object_path

    def create_bucket(self, name: str):
        logger.debug("Minio - Service - create_bucket")
        found = self._client.bucket_exists(name)
        if not found:
            self._client.make_bucket(name)

    def get_link(self, object_path: str, bucket_name: str = base_bucket) -> str:
        logger.debug("Minio - Service - get_link")
        url = self._client.get_presigned_url("GET", bucket_name, object_path)

        return url
