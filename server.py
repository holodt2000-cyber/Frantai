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

warnings.filterwarnings("ignore")

app = FastAPI(title="Frantai Smart Orchestrator")

# --- 1. КОНФИГУРАЦИЯ ---
KEYS = {
    "cerebras": os.getenv("CEREBRAS_KEY", "csk-x989hhdnn9p2nexmt24ndk9ten4k3cmd82je9k4jxcnjwh6x"),
    "sambanova": os.getenv("SAMBANOVA_KEY", "6460e865-5a60-4cd8-b854-07d8d991b344"),
    "groq": os.getenv("GROQ_KEY", "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"),
    "openrouter": os.getenv("OPENROUTER_KEY", "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")
}

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
    """Моментальный перевод через Groq"""
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="https://api.groq.com/openai/v1")
        prompt = f"Translate the following text to {target_lang}. Output ONLY the translation: {text}"
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], timeout=10
        ))
        return resp.choices[0].message.content.strip()
    except: return text

async def analyze_intent(text: str) -> str:
    """Улучшенный классификатор с жесткими правилами"""
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="https://api.groq.com/openai/v1")
        # Улучшаем промпт: добавляем описание категорий
        system_prompt = (
            "You are a classifier. Categories: "
            "CODE (programming, scripts, databases, html/css), "
            "MATH (formulas, equations), "
            "RESEARCH (news, facts, search), "
            "GENERAL (chat, greetings). "
            "Return ONLY the category name."
        )
        
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ], 
            timeout=5
        ))
        intent = resp.choices[0].message.content.strip().upper()
        # Чистим ответ от точек и лишних слов
        intent = re.sub(r'[^A-Z]', '', intent)
        return intent if intent in ["CODE", "MATH", "RESEARCH"] else "GENERAL"
    except: 
        return "GENERAL"

async def perplexity_search(query: str) -> str:
    """Метод Perplexity: поиск в реальном времени"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='ru-ru', max_results=3))
            return "\n\n".join([f"Source: {r['title']}\n{r['body']}" for r in results])
    except: return ""

def analyze_complexity(text: str) -> bool:
    """Выбор между Heavy и Fast очередью"""
    score = 0
    if len(text.split()) > 15: score += 2
    if re.search(r'[{}[\]()=;<>?]', text): score += 3
    if any(kw in text.lower() for kw in ['код', 'напиши', 'реши', 'почему']): score += 2
    return score >= 4

def get_family_by_intent(intent: str) -> str:
    """Маппинг задачи на семейство"""
    return {"CODE": "deepseek", "MATH": "llama", "RESEARCH": "deepseek"}.get(intent, "llama")

# --- 3. ЦЕНТРАЛЬНЫЙ ПРОЦЕССОР (CORE) ---

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    user_query = request.messages[-1].content
    intent = request.intent or await analyze_intent(user_query)
    
    # Режим Perplexity для RESEARCH
    if intent == "RESEARCH":
        context = await perplexity_search(user_query)
        if context:
            request.messages[-1].content += f"\n\nCONTEXT FROM WEB:\n{context}"

    f_name = request.family.lower()
    f_data = FAMILIES.get(f_name, FAMILIES['llama'])
    
    is_complex = analyze_complexity(user_query)
    queue = f_data["heavy"] + f_data["fast"] if is_complex else f_data["fast"] + f_data["heavy"]
    is_russian = bool(re.search('[а-яА-Я]', user_query))

    for target in queue:
        try:
            current_content = request.messages[-1].content
            
            # Перевод для Heavy моделей
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

            # Внутри блока try метода ask_ai, перед return:
            res = {
                "status": "success", 
                "model": resp.model, 
                "intent": intent,     # Это поможет нам понять, почему выбрана Llama
                "family": f_name, 
                "content": final_content
            }
            if request.thinking and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                res["thought"] = msg.reasoning_content
            
            return res
            
        except Exception as e:
            print(f"⚠️ Fail: {target['model']} | Error: {str(e)[:50]}")
            continue

    raise HTTPException(status_code=503, detail="Failover exhausted")

# --- 4. ЭКСПЕРТНЫЙ ПУТЬ (ИСПРАВЛЕНО НА POST) ---

@app.post("/expert")
async def ask_expert(request: ChatRequest):
    """Маршрутизация по типам задач"""
    intent = request.intent or await analyze_intent(request.messages[-1].content)
    request.family = get_family_by_intent(intent)
    request.intent = intent
    return await ask_ai(request)

# --- 5. СЛУЖЕБНЫЕ ЭНДПОИНТЫ ---

@app.get("/health")
@app.head("/health")
async def health(): return {"status": "online"}

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h2>Frantai Orchestrator Running</h2>"