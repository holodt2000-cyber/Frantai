import warnings
import os
import re
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from duckduckgo_search import DDGS
from fastapi.responses import HTMLResponse
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

warnings.filterwarnings("ignore")
app = FastAPI(title="Frantai Smart Orchestrator")

# --- 1. CONFIG ---
KEYS = {
    "cerebras": os.getenv("CEREBRAS_KEY", "csk-x989hhdnn9p2nexmt24ndk9ten4k3cmd82je9k4jxcnjwh6x"),
    "sambanova": os.getenv("SAMBANOVA_KEY", "6460e865-5a60-4cd8-b854-07d8d991b344"),
    "groq": os.getenv("GROQ_KEY", "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"),
    "openrouter": os.getenv("OPENROUTER_KEY", "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")
}
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "6796190792:AAEngQeCe0z7XtwhyqCpB-1ADWgBOXx9VWo")

FAMILIES = {
    "deepseek": {
        "heavy": [
            {"url": "https://api.sambanova.ai/v1", "key": KEYS["sambanova"], "model": "DeepSeek-R1", "heavy": True},
            {"url": "https://openrouter.ai/api/v1", "key": KEYS["openrouter"], "model": "deepseek/deepseek-r1:free", "heavy": True}
        ],
        "fast": [{"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "deepseek-r1-distill-llama-70b", "heavy": False}]
    },
    "llama": {
        "heavy": [
            {"url": "https://api.sambanova.ai/v1", "key": KEYS["sambanova"], "model": "Meta-Llama-3.1-405B-Instruct", "heavy": True},
            {"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "llama-3.3-70b-versatile", "heavy": True}
        ],
        "fast": [
            {"url": "https://api.cerebras.ai/v1", "key": KEYS["cerebras"], "model": "llama-3.3-70b", "heavy": False},
            {"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "gemma2-9b-it", "heavy": False}
        ]
    }
}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    family: str = "llama"
    thinking: bool = True
    intent: Optional[str] = None

# --- 2. MODULES ---

async def quick_translate(text: str, target_lang: str):
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="https://api.groq.com/openai/v1")
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": f"Translate to {target_lang}. ONLY translation text: {text}"}],
            timeout=10
        ))
        return resp.choices[0].message.content.strip()
    except: return text

async def smart_translate_back(text: str):
    if len(re.findall(r'[а-яА-Я]', text)) > len(text) * 0.15: return text
    parts = re.split(r'(```[\s\S]*?```)', text)
    translated = []
    for p in parts:
        if p.startswith('```'): translated.append(p)
        elif p.strip(): translated.append(await quick_translate(p, "Russian"))
        else: translated.append(p)
    return "".join(translated)

async def analyze_intent(text: str) -> str:
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)")
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages= [
            {
                "role": "system", 
                "content": (
                    "You are a strict classifier. Categorize user query into ONLY ONE word:\n"
                    "CODE: Programming, scripts, SQL, HTML/CSS, errors, architecture.\n"
                    "MATH: Calculations, logic puzzles, formulas.\n"
                    "RESEARCH: News, current events, fact-checking, web search.\n"
                    "GENERAL: Greetings, chat, or if no other category fits.\n\n"
                    "Examples:\n"
                    "'Write a python script' -> CODE\n"
                    "'How to fix 404 error' -> CODE\n"
                    "'2+2*2' -> MATH\n"
                    "'Latest bitcoin price' -> RESEARCH"
                )
            },
            {"role": "user", "content": text}
        ],
            temperature=0, timeout=7
        ))
        match = re.search(r'(CODE|MATH|RESEARCH|GENERAL)', resp.choices[0].message.content.upper())
        return match.group(0) if match else "GENERAL"
    except: return "GENERAL"

def get_family_by_intent(intent: str) -> str:
    return {"CODE": "deepseek", "MATH": "llama", "RESEARCH": "deepseek"}.get(intent, "llama")
# --- 3. CORE CORE ---

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    user_query = request.messages[-1].content
    intent = request.intent or await analyze_intent(user_query)
    
    f_data = FAMILIES.get(request.family.lower(), FAMILIES['llama'])
    is_russian = bool(re.search('[а-яА-Я]', user_query))
    queue = f_data["heavy"] + f_data["fast"] # Для экспертного режима всегда пробуем Heavy первым

    for target in queue:
        try:
            curr_msg = user_query
            if target["heavy"] and is_russian:
                curr_msg = await quick_translate(user_query, "English")
                curr_msg += "\n\nSTRICT: Answer in Russian. Keep all code, variables and technical comments in English inside blocks."

            client = OpenAI(api_key=target['key'].strip(), base_url=target['url'])
            req_messages = [m.model_dump() for m in request.messages]
            req_messages[-1]["content"] = curr_msg

            resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
                model=target['model'], messages=req_messages, timeout=90
            ))

            final_content = resp.choices[0].message.content
            if target["heavy"] and is_russian:
                final_content = await smart_translate_back(final_content)

            res = {"status": "success", "model": resp.model, "intent": intent, "content": final_content}
            if hasattr(resp.choices[0].message, 'reasoning_content'):
                res["thought"] = resp.choices[0].message.reasoning_content
            return res
        except: continue
    raise HTTPException(status_code=503, detail="Failover exhausted")

@app.post("/expert")
async def ask_expert(request: ChatRequest):
    intent = await analyze_intent(request.messages[-1].content)
    request.family = get_family_by_intent(intent)
    request.intent = intent
    return await ask_ai(request)

# --- 4. BOT ---

if TG_TOKEN:
    bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    dp = Dispatcher()

    async def safe_send(message: types.Message, text: str, prefix: str = ""):
        if not text: return
        limit = 3900
        parts = []
        while len(text) > 0:
            if len(text) <= limit:
                parts.append(text); break
            split = text.rfind('\n', 0, limit)
            if split <= 0: split = limit
            parts.append(text[:split])
            text = text[split:].lstrip()
        
        for i, p in enumerate(parts):
            label = f"{prefix} (Часть {i+1}/{len(parts)})" if len(parts) > 1 else prefix
            output = f"{label}\n\n{p}"
            try: await message.answer(output, parse_mode=ParseMode.MARKDOWN)
            except: await message.answer(output, parse_mode=None)

    @dp.message(Command("start"))
    async def cmd_start(m: types.Message): await m.answer("🚀 *Frantai Orchestrator Online*")

    @dp.message()
    async def handle_msg(m: types.Message):
        await bot.send_chat_action(m.chat.id, "typing")
        try:
            res = await ask_expert(ChatRequest(messages=[Message(role="user", content=m.text)]))
            if "thought" in res:
                await safe_send(m, res["thought"], "🧠 *Мысли модели:*")
            
            header = f"🤖 *Model:* `{res['model']}`\n🎯 *Intent:* `{res['intent']}`"
            await safe_send(m, res["content"], header)
        except Exception as e:
            await m.answer(f"❌ Ошибка: {str(e)[:100]}")

# --- 5. START ---
@app.on_event("startup")
async def startup():
    if TG_TOKEN:
        print("🤖 Перезапуск бота...")
        # Сначала удаляем вебхук и все зависшие сообщения
        await bot.delete_webhook(drop_pending_updates=True)
        # Небольшая пауза, чтобы Telegram успел закрыть старую сессию
        await asyncio.sleep(1) 
        asyncio.create_task(dp.start_polling(bot))

@app.get("/")
async def index(): return HTMLResponse("⚙️ Orchestrator Active")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))