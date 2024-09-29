from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from configs.Environment import get_environment_variables

env = get_environment_variables()

DATABASE_URL = f"postgresql+asyncpg://{env.POSTGRES_USER}:{env.POSTGRES_PASSWORD}@{env.POSTGRES_HOST}:{env.POSTGRES_PORT}/{env.POSTGRES_DB}"

engine = create_async_engine(DATABASE_URL, future=True)

async_session = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)


async def get_db_connection():
    async with async_session() as session:
        yield session
