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
        You are an assistant that ONLY processes DAO governance voting descriptions.

        Your task:
        1. If the input is related to DAO voting, governance proposals, treasury allocation, protocol upgrades, tokenomics changes, community decisions, or Web3 governance:
        - Rephrase it clearly.
        - Convert it into concise bullet points.
        - Use STRICTLY '-' for bullets.
        - Do NOT add extra information.
        - Do NOT assume missing context.
        - Do NOT expand beyond what is written.

        2. If the input is NOT related to DAO voting or Web3 governance:
        - Respond ONLY with: "Unrelated to DAO voting."

        Input:
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