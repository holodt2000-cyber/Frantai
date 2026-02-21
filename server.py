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

# Отключаем лишние ворнинги
warnings.filterwarnings("ignore")

app = FastAPI(title="Frantai Smart Orchestrator")

# --- 1. КОНФИГУРАЦИЯ (API КЛЮЧИ) ---
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
        "fast": [
            {"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "deepseek-r1-distill-llama-70b", "heavy": False}
        ]
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

# --- 2. ВСПОМОГАТЕЛЬНЫЕ МОДУЛИ (ПЕРЕВОД, ПОИСК, ИНТЕНТ) ---

async def quick_translate(text: str, target_lang: str):
    """Базовый перевод через быструю модель Groq"""
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="https://api.groq.com/openai/v1")
        prompt = f"Translate to {target_lang}. Output ONLY translation: {text}"
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], timeout=10
        ))
        return resp.choices[0].message.content.strip()
    except: return text

async def smart_translate_back(text: str):
    """Переводит текст на русский, НЕ трогая блоки кода ```...```"""
    if len(re.findall(r'[а-яА-Я]', text)) > len(text) * 0.15:
        return text
        
    parts = re.split(r'(```[\s\S]*?```)', text)
    translated_parts = []
    for part in parts:
        if part.startswith('```'):
            translated_parts.append(part)
        elif part.strip():
            translated_parts.append(await quick_translate(part, "Russian"))
        else:
            translated_parts.append(part)
    return "".join(translated_parts)

async def analyze_intent(text: str) -> str:
    """Классифицирует запрос пользователя"""
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)")
        messages = [
            {"role": "system", "content": "Categorize query: CODE, MATH, RESEARCH, GENERAL. Reply with ONE word."},
            {"role": "user", "content": text}
        ]
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=messages, temperature=0, timeout=7
        ))
        raw = resp.choices[0].message.content.strip().upper()
        match = re.search(r'(CODE|MATH|RESEARCH|GENERAL)', raw)
        return match.group(0) if match else "GENERAL"
    except: return "GENERAL"

async def perplexity_search(query: str) -> str:
    """Поиск в сети через DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='ru-ru', max_results=3))
            return "\n\n".join([f"Source: {r['title']}\n{r['body']}" for r in results])
    except: return ""

def analyze_complexity(text: str) -> bool:
    """Определяет сложность задачи для выбора Heavy/Fast модели"""
    score = 0
    if len(text.split()) > 15: score += 2
    if re.search(r'[{}[\]()=;<>?]', text): score += 3
    return score >= 3

def get_family_by_intent(intent: str) -> str:
    return {"CODE": "deepseek", "MATH": "llama", "RESEARCH": "deepseek"}.get(intent, "llama")

# --- 3. ЦЕНТРАЛЬНЫЙ ПРОЦЕССОР (CORE AI) ---

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    user_query = request.messages[-1].content
    intent = request.intent or await analyze_intent(user_query)
    
    if intent == "RESEARCH":
        context = await perplexity_search(user_query)
        if context: request.messages[-1].content += f"\n\nWEB CONTEXT:\n{context}"

    f_name = request.family.lower()
    f_data = FAMILIES.get(f_name, FAMILIES['llama'])
    is_complex = analyze_complexity(user_query)
    queue = f_data["heavy"] + f_data["fast"] if is_complex else f_data["fast"] + f_data["heavy"]
    is_russian = bool(re.search('[а-яА-Я]', user_query))

    for target in queue:
        try:
            current_content = request.messages[-1].content
            if target["heavy"] and is_russian:
                current_content = await quick_translate(current_content, "English")
                current_content += "\n\nSTRICT: Answer in Russian, but keep ALL code and variables in English blocks."

            client = OpenAI(api_key=target['key'].strip(), base_url=target['url'])
            req_messages = [m.model_dump() for m in request.messages]
            req_messages[-1]["content"] = current_content

            resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
                model=target['model'], messages=req_messages, timeout=85
            ))

            final_content = resp.choices[0].message.content
            
            if target["heavy"] and is_russian:
                final_content = await smart_translate_back(final_content)

            res = {"status": "success", "model": resp.model, "intent": intent, "content": final_content}
            if request.thinking and hasattr(resp.choices[0].message, 'reasoning_content'):
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

# --- 4. ТЕЛЕГРАМ БОТ (ИНТЕРФЕЙС) ---

if TG_TOKEN:
    bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start_h(m: types.Message):
        await m.answer("🚀 *Frantai Smart Orchestrator Active*\nЗадавай любой вопрос.")

    @dp.message()
    async def message_handler(message: types.Message):
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        
        try:
            req = ChatRequest(messages=[Message(role="user", content=message.text)])
            response = await ask_expert(req)
            
            content = response.get('content', '')
            thought = response.get('thought', '')
            header = f"🤖 *Model:* `{response['model']}`\n🎯 *Intent:* `{response['intent']}`\n\n"

            # Внутренняя функция для безопасной отправки длинных текстов
            async def safe_send(text, prefix=""):
                if not text: return
                
                # Лимит чуть меньше 4096 для запаса на разметку
                limit = 3900 
                
                # Разбивка текста на части
                chunks = []
                while len(text) > 0:
                    if len(text) <= limit:
                        chunks.append(text)
                        break
                    split_pos = text.rfind('\n', 0, limit)
                    if split_pos <= 0: split_pos = limit
                    chunks.append(text[:split_pos])
                    text = text[split_pos:].lstrip()

                # Отправка каждой части
                for i, chunk in enumerate(chunks):
                    msg_part = f"{prefix} (Часть {i+1})\n\n{chunk}" if len(chunks) > 1 else f"{prefix}\n\n{chunk}"
                    try:
                        await message.answer(msg_part, parse_mode=ParseMode.MARKDOWN)
                    except:
                        await message.answer(msg_part, parse_mode=None)

            # 1. Сначала отправляем мысли (если есть)
            if thought:
                await safe_send(thought, prefix="🧠 *Мысли модели:*")

            # 2. Затем отправляем основной контент с заголовком
            await safe_send(content, prefix=header)

        except Exception as e:
            # Выводим более детальную ошибку для отладки
            print(f"Error in handler: {e}")
            await message.answer(f"❌ Ошибка: {str(e)[:100]}")
# --- 5. ЗАПУСК И ЗДОРОВЬЕ ---

@app.on_event("startup")
async def on_startup():
    if TG_TOKEN:
        await bot.delete_webhook(drop_pending_updates=True)
        asyncio.create_task(dp.start_polling(bot))

@app.get("/")
async def root(): return HTMLResponse("<h2>Orchestrator + TG Bot Online</h2>")

@app.get("/health")
async def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Render использует переменную окружения PORT
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))