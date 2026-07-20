from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from database.config import get_settings

web_route = APIRouter()
settings = get_settings()
templates = Jinja2Templates(directory="view")


@web_route.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(request, "index.html")


@web_route.get("/sign-in", response_class=HTMLResponse)
async def sign_in_page(request: Request):
    return templates.TemplateResponse(request, "sign_in.html")


@web_route.get("/sign-up", response_class=HTMLResponse)
async def sign_up_page(request: Request):
    return templates.TemplateResponse(request, "sign_up.html")


@web_route.get("/patients/{patient_id}", response_class=HTMLResponse)
async def patient_page(request: Request, patient_id: str):
    return templates.TemplateResponse(request, "patient.html")
