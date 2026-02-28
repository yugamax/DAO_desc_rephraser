import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Groq Sentence Rephraser API")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Request model
class SentenceRequest(BaseModel):
    sentence: str


def rephrase_to_points(sentence: str) -> str:
    prompt = f"""
    Rephrase the following sentence clearly and convert it into concise bullet points.
    Format strictly using '-' for bullets.

    Sentence:
    "{sentence}"
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rephrases text clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


@app.post("/rephrase")
def rephrase_sentence(request: SentenceRequest):
    result = rephrase_to_points(request.sentence)

    return {
        "result": result
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)