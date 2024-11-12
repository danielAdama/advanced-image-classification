from fastapi import status, Request
import uuid
from datetime import datetime as dt
from typing import Union, Dict, Optional
from pathlib import Path
from src.schemas import image_schema
from src.exceptions.custom_exception import (
    APIAuthenticationFailException, InternalServerException, RecordNotFoundException
)
from vision import ChessCategorizer
from src.utils.app_utils import AppUtil
from src.utils.app_notification_message import NotificationMessage
from io import BytesIO
from PIL import Image
import base64

from config.logger import Logger

logger = Logger(__name__)

class ImageService:
    @staticmethod
    async def recognize(image: image_schema.ImageIn):
        try:
            # Decode the Base64 image string
            image_data = base64.b64decode(image.image)
            image_obj = Image.open(BytesIO(image_data)).convert('RGB')
            
            classifier = ChessCategorizer()
            result = classifier.classify(image_obj)

            logger.info("Image processed successfully")
        except Exception as ex:
            logger.error(f"Processing Image -> API v1/image/classify/: {ex}")
            raise InternalServerException()
        
        return result