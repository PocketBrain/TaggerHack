from minio import Minio
from configs.Environment import get_environment_variables

env = get_environment_variables()

base_bucket = env.MINIO_BASE_BUCKET

minio_client = Minio(
    env.MINIO_HOST,
    access_key=env.MINIO_ACCESS,
    secret_key=env.MINIO_SECRET,
    secure=True,
)


def get_minio_client() -> Minio:
    yield minio_client
