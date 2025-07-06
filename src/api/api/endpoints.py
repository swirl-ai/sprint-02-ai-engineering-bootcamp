from fastapi import APIRouter, Request
import logging

from api.rag.retrieval import rag_pipeline_wrapper

from api.api.models import RAGRequest, RAGResponse, RAGUsedImage


logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/rag")
async def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    result = rag_pipeline_wrapper(payload.query)
    used_image_urls = [RAGUsedImage(image_url=image["image_url"], price=image["price"], description=image["description"]) for image in result["retrieved_images"]]

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        used_image_urls=used_image_urls
    )


api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])