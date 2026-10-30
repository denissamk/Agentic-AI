from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# Load the GOOGLE_API_KEY from the .env file so we can call Gemini.
load_dotenv()
client = Groq()

# This one line creates our web app. We will hang all our endpoints on it.
app = FastAPI()


# -------------------------------------------------------------------
# Part 3: our first endpoint (a simple GET)
# Open http://127.0.0.1:8000/ in the browser to see this.
# -------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Welcome, Kaise ho aap?"}


# -------------------------------------------------------------------
# Part 4: an endpoint with inputs (query parameters)
# Try: http://127.0.0.1:8000/margin?revenue=5000&expenses=3000
# Notice the type hints (revenue: float). FastAPI uses them to read the
# inputs from the URL and turn the text into numbers for us.
# -------------------------------------------------------------------
@app.get("/margin")
def margin(revenue: float, expenses: float):
    profit = revenue - expenses
    margin_percent = (profit / revenue) * 100
    return {"profit": profit, "margin_percent": margin_percent}


# -------------------------------------------------------------------
# Part 5 + 6: a POST endpoint that calls the AI
# The user sends a review, we ask Gemini for the sentiment, and we send
# back a small, clean result. Low output = few tokens used.
# -------------------------------------------------------------------

# What the user must SEND us (the request body)
class Review(BaseModel):
    text: str


# What Gemini must give back, and what we SEND to the user (the response)
class Sentiment(BaseModel):
    label: str   # "positive", "negative", or "neutral"
    score: int   # 1 (very bad) to 5 (very good)


@app.post("/sentiment")
def analyze_sentiment(review: Review):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze customer review sentiment. Reply only with JSON: "
                    '{"label": "positive" | "negative" | "neutral", "score": 1-5}. '
                    "score is 1 (very bad) to 5 (very good)."
                ),
            },
            {"role": "user", "content": f"Review: {review.text}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "sentiment", "schema": Sentiment.model_json_schema()},
        },
        temperature=0,
    )
    # Groq returns a JSON string, so validate it into your Sentiment model.
    return Sentiment.model_validate_json(response.choices[0].message.content)
