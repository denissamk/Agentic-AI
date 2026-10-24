# Lesson 10: Build Your Own AI API (uv + FastAPI)

So far, every AI feature we built runs only on **our** machine, inside a notebook or a Streamlit app. But what if a mobile app, a website, or another team wants to use our AI?

We cannot send them our notebook. We need to put our AI behind a **web address (URL)** that anyone can call. That is what an **API** is.

In this lesson we will build a small AI service: you send it a customer review, and it sends back the sentiment (positive / negative / neutral) with a score from 1 to 5.

We will use two tools:
- **uv** - a fast tool to create Python projects and install packages.
- **FastAPI** - a tool to turn our Python functions into a web API.

---

## Part 1: What is an API? (the "why")

Think of a restaurant.

- You (the customer) do not go into the kitchen and cook.
- You give your order to a **waiter**.
- The waiter takes it to the kitchen and brings back your food.

An **API is the waiter**. Other programs send a request to our API, our API does the work (calls Gemini), and sends back the answer. The other program never sees our code or our API key. It just gets the food.

FastAPI is a popular, beginner-friendly way to build this "waiter" in Python.

---

## Part 2: Set up the project with uv

Until now we used `pip install`. From here we use **uv**, which is much faster and keeps each project clean and separate.

First, create the project:

```bash
uv init 10_fastapi
cd 10_fastapi
```

This makes a folder with a `pyproject.toml` file. Think of `pyproject.toml` as a shopping list - it remembers every package this project needs.

Now add the packages we need:

```bash
uv add "fastapi[standard]" google-genai python-dotenv pydantic
```

That's it. uv created a private environment for this project and installed everything. No `pip`, no `venv` commands to remember.

> Tip: `fastapi[standard]` installs FastAPI **plus** the web server that runs it. The square brackets just mean "give me the full package, with the extras."

---

## Part 3: Your first endpoint

An **endpoint** is one URL that does one job. Let's make the simplest one.

Create a file `main.py`:

```python
from fastapi import FastAPI

# This one line creates our web app.
app = FastAPI()

@app.get("/")
def home():
    return {"message": "My AI API is running!"}
```

Two new things:
- `app = FastAPI()` creates the app (remember objects from the classes lesson? `app` is an object).
- `@app.get("/")` sits on top of a function. It tells FastAPI: "when someone visits the address `/`, run this function." The `/` is the home address.

Now run the server:

```bash
uv run fastapi dev main.py
```

Open **http://127.0.0.1:8000/** in your browser. You will see your message. Congratulations, you just put Python on the web!

> `127.0.0.1` always means "this computer." The server keeps running and reloads automatically every time you save the file. Press `Ctrl + C` to stop it.

---

## Part 4: An endpoint that takes inputs

A waiter that ignores your order is useless. Let's take some inputs.

We'll reuse our business example - profit margin. Add this to `main.py`:

```python
@app.get("/margin")
def margin(revenue: float, expenses: float):
    profit = revenue - expenses
    margin_percent = (profit / revenue) * 100
    return {"profit": profit, "margin_percent": margin_percent}
```

Now open:

```
http://127.0.0.1:8000/margin?revenue=5000&expenses=3000
```

You get back `{"profit": 2000.0, "margin_percent": 40.0}`.

The part after `?` are the **inputs** (called query parameters). Notice `revenue: float` - this is the **type hint** we learned earlier. It is finally doing real work here: FastAPI reads `5000` from the URL (which is just text) and turns it into a number for us, because we told it `float`. Type hints are not decoration anymore - FastAPI depends on them.

---

## Part 5: Sending bigger data with POST

A URL is fine for two small numbers. But you cannot put a long paragraph of customer feedback into a URL nicely. For bigger data we use **POST**, and we send the data in a clean package called the **request body**.

To describe what that package should look like, we use **Pydantic** (from the classes lesson):

```python
from pydantic import BaseModel

class Review(BaseModel):
    text: str
```

This says: "anyone calling this endpoint must send a `text` field, and it must be a string." If they send something wrong, FastAPI rejects it automatically with a clear error. We get free validation, just like in the structured-output lesson.

---

## Part 6: The AI payoff

Now we connect everything to Gemini. The full `main.py` is in this folder. Here is the AI part:

```python
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# What Gemini gives back, and what we send to the user:
class Sentiment(BaseModel):
    label: str   # "positive", "negative", or "neutral"
    score: int   # 1 (very bad) to 5 (very good)

@app.post("/sentiment")
def analyze_sentiment(review: Review):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Find the sentiment of this customer review. "
            "label must be 'positive', 'negative', or 'neutral'. "
            "score must be a number from 1 (very bad) to 5 (very good). "
            f"Review: {review.text}"
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Sentiment,
        ),
    )
    return response.parsed
```

We kept the answer tiny on purpose - just a label and a score. Small answers use **fewer tokens**, which is friendly to the Gemini free tier. You can test it many times without worry.

> Put your key in a `.env` file in this folder: `GOOGLE_API_KEY=your_key_here`. See `sample.env`.

---

## The best part: free testing page

You do not need any extra tool to test a POST endpoint. While the server is running, open:

**http://127.0.0.1:8000/docs**

FastAPI built a full testing page for you - automatically. Click `/sentiment`, click "Try it out", type a review, and hit Execute. Try:

> "The delivery was late and the food was cold."

You'll get back something like `{"label": "negative", "score": 1}`. Your AI is now a real web service.

---

## Exercise

Add a new endpoint `/tagline` to the same app.

- It should be a **POST** endpoint.
- The user sends a product name (for example, "masala chai").
- Your API asks Gemini for a short, catchy one-line tagline and sends it back.

Hints:
- Make a Pydantic model for the input (a `product` field).
- Keep the output short - one line - to save tokens.
- Test it from the `/docs` page.

Bonus: also return a single emoji that matches the product.

---

That's it - you can now wrap any AI feature as an API and share it with the world. In the bonus ML track, we'll reuse this exact pattern to serve a machine learning model (Lesson 15).
