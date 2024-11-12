from fastapi import APIRouter, status, Query, Depends, Path, Request
import uuid
from src.schemas import image_schema
from src.services import image_service
from typing import Dict, Union, List, Annotated
from src.utils.request_response import ApiResponse

image_router = APIRouter(prefix="/image", tags=["Image"])

@image_router.post(
        '/classify/',
        response_model=Dict[str, Union[str]],
        status_code=status.HTTP_201_CREATED
    )
async def send_image(
    image: image_schema.ImageIn
    ):
    result = await image_service.ImageService.recognize(image)
    return ApiResponse(
        message="Image added successfully",
        data=result
    )