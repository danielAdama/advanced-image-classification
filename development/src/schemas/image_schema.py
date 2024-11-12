from pydantic import BaseModel, Field
from fastapi import UploadFile
from typing import List, Dict
from datetime import datetime as dt
from uuid import UUID, uuid4
from typing import Optional, Union
from src.utils.base import BaseSchema

class ImageIn(BaseModel):
    file: UploadFile