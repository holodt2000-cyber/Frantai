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
app = FastAPI(title="Frantai Smart Orchestrator")
# --- 1. КОНФИГУРАЦИЯ ---
KEYS = {
    "cerebras": os.getenv("CEREBRAS_KEY", "csk-x989hhdnn9p2nexmt24ndk9ten4k3cmd82je9k4jxcnjwh6x"),
    "sambanova": os.getenv("SAMBANOVA_KEY", "6460e865-5a60-4cd8-b854-07d8d991b344"),
    "groq": os.getenv("GROQ_KEY", "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"),
    "openrouter": os.getenv("OPENROUTER_KEY", "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")
}

# Добавь токен в переменные окружения на Render
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

# --- 2. ВСПОМОГАТЕЛЬНЫЕ МОДУЛИ ---

async def quick_translate(text: str, target_lang: str):
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="https://api.groq.com/openai/v1")
        prompt = f"Translate to {target_lang}. Output ONLY translation: {text}"
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], timeout=10
        ))
        return resp.choices[0].message.content.strip()
    except: return text

async def analyze_intent(text: str) -> str:
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="https://api.groq.com/openai/v1")
        
        # Усиленный промпт с примерами (Few-Shot)
        messages = [
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
        ]
        
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=messages, 
            temperature=0, 
            timeout=7
        ))
        
        # Берем только первое слово, убираем мусор
        raw_intent = resp.choices[0].message.content.strip().upper()
        intent = re.search(r'(CODE|MATH|RESEARCH|GENERAL)', raw_intent)
        
        return intent.group(0) if intent else "GENERAL"
    except:
        return "GENERAL"

async def perplexity_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='ru-ru', max_results=3))
            return "\n\n".join([f"Source: {r['title']}\n{r['body']}" for r in results])
    except: return ""

def analyze_complexity(text: str) -> bool:
    score = 0
    if len(text.split()) > 12: score += 2
    if re.search(r'[{}[\]()=;<>?]', text): score += 3
    if any(kw in text.lower() for kw in ['код', 'напиши', 'реши', 'сервер']): score += 2
    return score >= 3

def get_family_by_intent(intent: str) -> str:
    return {"CODE": "deepseek", "MATH": "llama", "RESEARCH": "deepseek"}.get(intent, "llama")

# --- 3. ЦЕНТРАЛЬНЫЙ ПРОЦЕССОР (CORE) ---

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    user_query = request.messages[-1].content
    intent = request.intent or await analyze_intent(user_query)
    
    if intent == "RESEARCH":
        context = await perplexity_search(user_query)
        if context:
            request.messages[-1].content += f"\n\nWEB CONTEXT:\n{context}"

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

            client = OpenAI(api_key=target['key'].strip(), base_url=target['url'])
            req_messages = [m.model_dump() for m in request.messages]
            req_messages[-1]["content"] = current_content

            resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
                model=target['model'], messages=req_messages, timeout=85
            ))

            msg = resp.choices[0].message
            final_content = msg.content

            if target["heavy"] and is_russian:
                final_content = await quick_translate(final_content, "Russian")

            res = {
                "status": "success", "model": resp.model, 
                "intent": intent, "family": f_name, "content": final_content
            }
            if request.thinking and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                res["thought"] = msg.reasoning_content
            
            return res
        except: continue

    raise HTTPException(status_code=503, detail="Failover exhausted")

@app.post("/expert")
async def ask_expert(request: ChatRequest):
    intent = request.intent or await analyze_intent(request.messages[-1].content)
    request.family = get_family_by_intent(intent)
    request.intent = intent
    return await ask_ai(request)

# --- 4. TELEGRAM BOT INTEGRATION ---

if TG_TOKEN:
    bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start_handler(message: types.Message):
        await message.answer("🚀 *Frantai Orchestrator Online*\nОтправьте любой запрос.")

    @dp.message()
    async def message_handler(message: types.Message):
        # Эмуляция ChatRequest для вызова внутренней логики
        req = ChatRequest(messages=[Message(role="user", content=message.text)])
        
        # Отправляем статус "печатает"
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        
        try:
            # Вызываем экспертную логику напрямую
            response = await ask_expert(req)
            
            header = f"🤖 *Model:* `{response['model']}` | 🎯 *Intent:* `{response['intent']}`\n\n"
            content = response['content']
            
            # Если ответ слишком длинный для одного сообщения (лимит 4096)
            full_text = header + content
            if len(full_text) > 4000:
                for i in range(0, len(full_text), 4000):
                    await message.answer(full_text[i:i+4000])
            else:
                await message.answer(full_text)
        except Exception as e:
            await message.answer(f"⚠️ Ошибка: {str(e)[:100]}")

# --- 5. LIFECYCLE & UTILS ---

@app.on_event("startup")
async def on_startup():
    if TG_TOKEN:
        print("✅ Telegram Bot polling started...")
        asyncio.create_task(dp.start_polling(bot))

@app.get("/health")
async def health(): return {"status": "online"}

@app.get("/", response_class=HTMLResponse)
async def root(): return "<h2>Orchestrator + TG Bot Active</h2>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))