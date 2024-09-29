from fastapi import FastAPI
from loguru import logger
from starlette import status

from fastapi import Request
from fastapi.responses import JSONResponse

from errors.errors import (
    ErrEntityNotFound,
    ErrEntityConflict,
    ErrBadRequest,
    ErrNotAuthorized,
)


async def entity_not_found_exception_handler(request: Request, e: ErrEntityNotFound):
    logger.debug(f"err = {e}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(e)}
    )


async def entity_conflict_exception_handler(request: Request, e: ErrEntityConflict):
    logger.debug(f"err = {e}")
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT, content={"detail": str(e)}
    )


async def not_authorized_exception_handler(request: Request, e: ErrNotAuthorized):
    logger.debug(f"err = {e}")
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": str(e)}
    )


async def forbidden_exception_handler(request: Request, e: ErrNotAuthorized):
    logger.debug(f"err = {e}")
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(e)}
    )


async def bad_request_exception_handler(request: Request, e: ErrBadRequest):
    logger.debug(f"err = {e}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(e)}
    )


async def internal_server_exception_handler(request: Request, e: ErrBadRequest):
    logger.error(f"err = {e}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(e)}
    )


def init_exception_handlers(app: FastAPI):
    app.add_exception_handler(ErrEntityNotFound, entity_not_found_exception_handler)

    app.add_exception_handler(ErrEntityConflict, entity_conflict_exception_handler)

    app.add_exception_handler(ErrNotAuthorized, not_authorized_exception_handler)

    app.add_exception_handler(ErrBadRequest, bad_request_exception_handler)

    app.add_exception_handler(500, internal_server_exception_handler)
