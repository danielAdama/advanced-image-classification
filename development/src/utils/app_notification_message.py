from pydantic import BaseModel


class NotificationMessage:
    RETRIEVED_SUCCESSFULLY = "Retrieved successfully"
    RECORD_NOT_FOUND = "Record with specified constraints not found"
    INTERNAL_SERVER_ERROR = "Could not complete the request at the moment"