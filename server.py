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

app = FastAPI(title="Zero-Cost AI API", description="FastAPI wrapper for multiple AI providers")

# --- КОНФИГУРАЦИЯ ---
KEYS = {
    "cerebras": "csk-x989hhdnn9p2nexmt24ndk9ten4k3cmd82je9k4jxcnjwh6x",
    "sambanova": "6460e865-5a60-4cd8-b854-07d8d991b344",
    "groq": "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",
    "openrouter": "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"
}

FAMILIES = {
    "deepseek": {
        "heavy": [
            {"url": "https://api.sambanova.ai/v1", "key": KEYS["sambanova"], "model": "DeepSeek-R1"},
            {"url": "https://openrouter.ai/api/v1", "key": KEYS["openrouter"], "model": "deepseek/deepseek-r1:free"}
        ],
        "fast": [
            {"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "deepseek-r1-distill-llama-70b"},
            {"url": "https://api.cerebras.ai/v1", "key": KEYS["cerebras"], "model": "llama-3.3-70b"}
        ]
    },
    "llama": {
        "heavy": [
            {"url": "https://api.sambanova.ai/v1", "key": KEYS["sambanova"], "model": "Meta-Llama-3.1-405B-Instruct"},
            {"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "llama-3.3-70b-versatile"}
        ],
        "fast": [
            {"url": "https://api.cerebras.ai/v1", "key": KEYS["cerebras"], "model": "llama-3.3-70b"}, 
            {"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "gemma2-9b-it"}
        ]
    },
    "qwen": {
        "heavy": [{"url": "https://openrouter.ai/api/v1", "key": KEYS["openrouter"], "model": "qwen/qwen-2.5-72b-instruct:free"}],
        "fast": [{"url": "https://api.groq.com/openai/v1", "key": KEYS["groq"], "model": "qwen-2.5-32b"}]
    }
}

# --- МОДЕЛИ ДАННЫХ ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    family: str = "llama"
    thinking: bool = True

# --- ЛОГИКА ---
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
    
    is_heavy = analyze_complexity(user_query)
    queue = f_data["heavy"] + f_data["fast"] if is_heavy else f_data["fast"] + f_data["heavy"]
    queue.append({"url": "https://openrouter.ai/api/v1", "key": KEYS["openrouter"], "model": "openrouter/auto:free"})

    # Поиск (DuckDuckGo) - выносим в отдельный поток, чтобы не блочить async
    try:
        loop = asyncio.get_event_loop()
        with DDGS() as ddgs:
            res = await loop.run_in_executor(None, lambda: list(ddgs.text(user_query, region='ru-ru', max_results=1)))
            if res:
                request.messages[-1].content += f"\n\nКонтекст: {res[0]['body']}"
    except:
        pass

    for target in queue:
        try:
            client = OpenAI(api_key=target['key'].strip(), base_url=target['url'])
            headers = {"HTTP-Referer": "https://render.com"} if "openrouter" in target['url'] else None

            # Запрос к API (используем асинхронную обертку для синхронного клиента OpenAI)
            resp = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: client.chat.completions.create(
                    model=target['model'],
                    messages=[m.model_dump() for m in request.messages],
                    timeout=50,
                    extra_headers=headers
                )
            )

            final_model = resp.model
            msg = resp.choices[0].message
            
            result = {
                "status": "success",
                "model": f"{final_model} (via {target['model']})" if "auto" in target['model'] else final_model,
                "content": msg.content
            }

            if request.thinking and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                result["thought"] = msg.reasoning_content
            
            return result
            
        except Exception as e:
            print(f"⚠️ Fail {target['model']}: {str(e)[:50]}")
            continue

    raise HTTPException(status_code=503, detail="All providers failed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)