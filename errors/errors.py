class ErrEntityNotFound(Exception):
    def __int__(self, message):
        super().__init__(message)


class ErrEntityConflict(Exception):
    def __int__(self, message):
        super().__init__(message)


class ErrBadRequest(Exception):
    def __int__(self, message):
        super().__init__(message)


class ErrNotAuthorized(Exception):
    def __int__(self, message):
        super().__init__(message)


class ErrInternalServer(Exception):
    def __int__(self, message):
        super().__init__(message)
