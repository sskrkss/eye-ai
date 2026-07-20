from fastapi import APIRouter, status

from dto.response.message_response_dto import MessageResponseDto

health_check_route = APIRouter()


@health_check_route.get(
    "/health-check",
    response_model=MessageResponseDto,
    status_code=status.HTTP_200_OK,
    summary="App's health check"
)
async def health_check() -> MessageResponseDto:
    return MessageResponseDto(message="Health check successful")
