import warnings
import os
import re
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from duckduckgo_search import DDGS

warnings.filterwarnings("ignore")

app = FastAPI(title="Zero-Cost AI API with Translation")

# --- КОНФИГУРАЦИЯ ---
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
            {"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "deepseek-r1-distill-llama-70b", "heavy": False},
            {"url": "https://api.cerebras.ai/v1", "key": KEYS["cerebras"], "model": "llama-3.3-70b", "heavy": False}
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

# Вспомогательная функция для быстрого перевода через "быструю" модель
async def quick_translate(text: str, target_lang: str):
    try:
        client = OpenAI(api_key=KEYS["groq"], base_url="https://api.groq.com/openai/v1")
        prompt = f"Translate the following text to {target_lang}. Output ONLY the translation: {text}"
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.chat.completions.create(
                model="llama-3.3-70b-specdec", # Используем самую быструю модель для перевода
                messages=[{"role": "user", "content": prompt}],
                timeout=10
            )
        )
        return resp.choices[0].message.content.strip()
    except:
        return text # Если перевод упал, возвращаем оригинал

def analyze_complexity(text: str) -> bool:
    score = 0
    if len(text.split()) > 12: score += 2
    if re.search(r'[{}[\]()=;<>?]', text): score += 3
    if any(kw in text.lower() for kw in ['код', 'напиши', 'алгоритм', 'создай']): score += 2
    return score >= 4

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    user_query = request.messages[-1].content
    f_name = request.family.lower()
    f_data = FAMILIES.get(f_name, FAMILIES['llama'])
    
    is_complex = analyze_complexity(user_query)
    queue = f_data["heavy"] + f_data["fast"] if is_complex else f_data["fast"] + f_data["heavy"]
    
    # Пытаемся определить язык (упрощенно: если есть кириллица — русский)
    is_russian = bool(re.search('[а-яА-Я]', user_query))

    for target in queue:
        try:
            current_content = user_query
            
            # 1. Перевод на английский для HEAVY моделей
            if target["heavy"] and is_russian:
                current_content = await quick_translate(user_query, "English")

            client = OpenAI(api_key=target['key'].strip(), base_url=target['url'])
            
            # Подменяем последний месседж на переведенный (если нужно)
            request_messages = [m.model_dump() for m in request.messages]
            request_messages[-1]["content"] = current_content

            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.chat.completions.create(
                    model=target['model'],
                    messages=request_messages,
                    timeout=60
                )
            )

            msg = resp.choices[0].message
            final_content = msg.content

            # 2. Перевод обратно на русский
            if target["heavy"] and is_russian:
                final_content = await quick_translate(final_content, "Russian")

            result = {
                "status": "success",
                "model": resp.model,
                "content": final_content
            }

            if target["heavy"] and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                result["thought"] = msg.reasoning_content
            
            return result
            
        except Exception as e:
            print(f"⚠️ Ошибка {target['model']}: {str(e)[:50]}")
            continue

    raise HTTPException(status_code=503, detail="All providers failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))