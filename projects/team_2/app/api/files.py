import asyncio
import json
import os
import re
import secrets
from asyncio import create_task
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import cast, Callable, Optional, Awaitable
from src.summarization import summarize
from sentence_transformers import SentenceTransformer


import requests
from fastapi import (APIRouter, Depends, File, HTTPException, Request,
                     UploadFile, status)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

templates = Jinja2Templates(directory="templates")
model = SentenceTransformer('all-MiniLM-L6-v2')

router = APIRouter()


@router.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse(
        "general_pages/index.html",
        {"request": request}
    )


@router.get("/api/summarize/", response_class=JSONResponse)
async def summarize_article(article: str):
    sentences = _prepare_text(article)
    summarization = summarize(sentences, model=model, n_iter=250, length=4, capacity=.1)
    return {'summarization': summarization}


def _prepare_text(text: str):
    text = text.strip()
    sentences = re.split(r"[.?!;][ ']", text)
    return sentences


