import asyncio
import logging
import torch

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

import streamlit as st

from model import generate_comment
from youtube_api import get_prompt_for_model

torch.classes.__path__ = []

TOKEN = st.secrets["BOT_TOKEN"]

GREETING = "Hi! I am Commentator bot."
HELP = "Send me a YouTube video's URL and I will generate a comment"
UNIVERSAL_COMMENT = "Who is watching in 2025?"
ERROR_MESSAGE = "Error: URL is incorrect or video is unavailable. /help"

dp = Dispatcher()

@dp.message(Command("start"))
async def command_start_handler(message: Message) -> None:
    await message.answer(GREETING + "\n\n" + HELP)

@dp.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    await message.answer(HELP)

@dp.message()
async def url_handler(message: Message) -> None:
    prompt = get_prompt_for_model(message.text)
    if prompt:
        comment = generate_comment(prompt)
        await message.answer(comment)
    else:
        await message.answer(ERROR_MESSAGE)

async def main() -> None:
    bot = Bot(token=TOKEN)
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
