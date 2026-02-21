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

from fastapi.responses import HTMLResponse

@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "online"}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Frantai AI Chat</title>
        <style>
            body { background: #1a1a1a; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; height: 100vh; margin: 0; }
            #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
            .message { max-width: 80%; padding: 12px 16px; border-radius: 15px; line-height: 1.5; word-wrap: break-word; }
            .user { align-self: flex-end; background: #2b5278; color: white; border-bottom-right-radius: 2px; }
            .ai { align-self: flex-start; background: #333; border-bottom-left-radius: 2px; border: 1px solid #444; }
            .thought { font-style: italic; color: #888; font-size: 0.9em; border-left: 2px solid #555; padding-left: 10px; margin-bottom: 8px; }
            #input-area { background: #252525; padding: 20px; display: flex; gap: 10px; border-top: 1px solid #333; }
            input { flex: 1; background: #333; border: 1px solid #444; color: white; padding: 12px; border-radius: 8px; outline: none; }
            button { background: #0078d4; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; transition: 0.3s; }
            button:hover { background: #005a9e; }
            button:disabled { background: #555; cursor: not-allowed; }
            .model-info { font-size: 0.7em; color: #666; margin-top: 5px; text-align: right; }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div class="message ai">Привет! Я Frantai AI. Напиши что-нибудь, и я постараюсь помочь.</div>
        </div>
        <div id="input-area">
            <input type="text" id="user-input" placeholder="Введите ваш вопрос..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button id="send-btn" onclick="sendMessage()">Отправить</button>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('user-input');
                const btn = document.getElementById('send-btn');
                const container = document.getElementById('chat-container');
                const text = input.value.trim();

                if (!text) return;

                // Добавляем сообщение пользователя
                const userDiv = document.createElement('div');
                userDiv.className = 'message user';
                userDiv.textContent = text;
                container.appendChild(userDiv);

                input.value = '';
                btn.disabled = true;
                container.scrollTop = container.scrollHeight;

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            messages: [{role: "user", content: text}],
                            family: "llama",
                            thinking: true
                        })
                    });

                    if (!response.ok) throw new Error('Ошибка сервера');

                    const data = await response.json();
                    const aiDiv = document.createElement('div');
                    aiDiv.className = 'message ai';

                    if (data.thought) {
                        const thoughtDiv = document.createElement('div');
                        thoughtDiv.className = 'thought';
                        thoughtDiv.innerHTML = "<b>Рассуждение:</b><br>" + data.thought.replace(/\n/g, '<br>');
                        aiDiv.appendChild(thoughtDiv);
                    }

                    const contentDiv = document.createElement('div');
                    contentDiv.innerHTML = data.content.replace(/\n/g, '<br>');
                    aiDiv.appendChild(contentDiv);

                    const infoDiv = document.createElement('div');
                    infoDiv.className = 'model-info';
                    infoDiv.textContent = data.model || "AI Model";
                    aiDiv.appendChild(infoDiv);

                    container.appendChild(aiDiv);
                } catch (e) {
                    const errDiv = document.createElement('div');
                    errDiv.className = 'message ai';
                    errDiv.style.color = '#ff6b6b';
                    errDiv.textContent = 'Ошибка: не удалось получить ответ от сервера.';
                    container.appendChild(errDiv);
                } finally {
                    btn.disabled = false;
                    container.scrollTop = container.scrollHeight;
                }
            }
        </script>
    </body>
    </html>
    """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))