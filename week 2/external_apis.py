"""
External APIs as Tools-external_apis

Integrated REAL public REST APIs as agent tools (not mock data),
and handle API errors INSIDE each tool implementation so the agent
never crashes on a bad city name, bad currency code, missing API key,
rate limit, or network hiccup.

APIs used:
    1. get_weather        -> Open-Meteo (free, no API key needed)
    2. convert_currency   -> Frankfurter.app (free, no API key needed)
    3. get_news             -> NewsAPI.org (free tier, NEEDS an API key)

SETUP
-----
1. pip install requests groq python-dotenv
2. Get a free NewsAPI key at https://newsapi.org/register
3. Add to your .env file:
       GROQ_API_KEY=your_groq_key
       NEWS_API_KEY=your_newsapi_key
   If NEWS_API_KEY is missing, get_news() returns a clear error dict
   instead of crashing.
"""

import os
import time
import json

import requests
from dotenv import load_dotenv
from groq import Groq, BadRequestError

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

REQUEST_TIMEOUT = 8  # seconds - don't let a hung API freeze the agent


"""
Generic HTTP Retry Helper
--------------------------
Server-side / transient errors (429 rate limit, 500/502/503/504,
timeouts, connection drops) are worth retrying with backoff, since
they're often temporary.

Client errors (400/401/404 - bad key, bad city, bad currency) are NOT
retried, because retrying an inherently wrong request will never help.
"""

def http_get_with_retry(url: str, params: dict = None, max_retries: int = 2):
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code in (429, 500, 502, 503, 504):
                wait = 1.5 * (attempt + 1)
                print(f"    [http retry {attempt + 1}/{max_retries}] "
                      f"Got {response.status_code}, waiting {wait}s...")
                time.sleep(wait)
                last_exception = f"HTTP {response.status_code}"
                continue

            return response  # any other status (200, 400, 401, 404...) - caller handles it

        except (requests.ConnectionError, requests.Timeout) as e:
            last_exception = e
            wait = 1.5 * (attempt + 1)
            print(f"    [http retry {attempt + 1}/{max_retries}] "
                  f"Network error ({type(e).__name__}), waiting {wait}s...")
            time.sleep(wait)

    return None  # every attempt failed at the network level


"""
Tool 1 - Weather (real API, no key needed)
Open-Meteo needs coordinates, not city names, so this does two calls:
geocode the city name -> coordinates, then coordinates -> forecast.
"""

def get_weather(city: str) -> dict:

    geo_response = http_get_with_retry(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    )

    if geo_response is None:
        return {"error": f"Weather service unreachable (network issue) while looking up '{city}'"}

    if geo_response.status_code != 200:
        return {"error": f"Geocoding failed with status {geo_response.status_code}"}

    geo_data = geo_response.json()
    if not geo_data.get("results"):
        return {"error": f"Could not find a city named '{city}'"}

    place = geo_data["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    weather_response = http_get_with_retry(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,relative_humidity_2m"
        }
    )

    if weather_response is None:
        return {"error": "Weather service unreachable (network issue) fetching forecast"}

    if weather_response.status_code != 200:
        return {"error": f"Forecast request failed with status {weather_response.status_code}"}

    current = weather_response.json().get("current", {})

    return {
        "city": place.get("name", city),
        "country": place.get("country", "unknown"),
        "temperature_c": current.get("temperature_2m"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m")
    }


"""
Tool 2 - Currency Conversion (real API, no key needed)
Frankfurter.app is a free, keyless exchange-rate API backed by the ECB.
"""

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:

    from_curr = from_currency.upper().strip()
    to_curr = to_currency.upper().strip()

    response = http_get_with_retry(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_curr, "to": to_curr}
    )

    if response is None:
        return {"error": "Currency service unreachable (network issue)"}

    if response.status_code == 404:
        return {"error": f"Unsupported currency code: '{from_curr}' or '{to_curr}'"}

    if response.status_code != 200:
        return {"error": f"Currency conversion failed with status {response.status_code}"}

    data = response.json()
    rates = data.get("rates", {})

    if to_curr not in rates:
        return {"error": f"Could not get a rate for '{to_curr}'"}

    return {
        "amount": amount,
        "from": from_curr,
        "to": to_curr,
        "converted": round(rates[to_curr], 2),
        "date": data.get("date")
    }


"""
Tool 3 - News Search (real API, NEEDS a free key from newsapi.org)
"""

def get_news(topic: str, max_results: int = 5) -> dict:

    if not NEWS_API_KEY:
        return {
            "error": "NEWS_API_KEY is not set. Get a free key at "
                     "https://newsapi.org/register and add it to your .env file."
        }

    response = http_get_with_retry(
        "https://newsapi.org/v2/everything",
        params={
            "q": topic,
            "sortBy": "publishedAt",
            "pageSize": max_results,
            "language": "en",
            "apiKey": NEWS_API_KEY
        }
    )

    if response is None:
        return {"error": "News service unreachable (network issue)"}

    if response.status_code == 401:
        return {"error": "NEWS_API_KEY is invalid or expired. Check your .env file."}

    if response.status_code == 429:
        return {"error": "NewsAPI rate limit hit. Free tier allows limited requests per day."}

    if response.status_code != 200:
        return {"error": f"News search failed with status {response.status_code}"}

    data = response.json()
    articles = data.get("articles", [])

    if not articles:
        return {"topic": topic, "found": 0, "articles": []}

    cleaned = []
    for article in articles:
        cleaned.append({
            "title": article.get("title"),
            "source": (article.get("source") or {}).get("name"),
            "published_at": article.get("publishedAt"),
            "description": article.get("description") or "",
            "url": article.get("url")
        })

    return {"topic": topic, "found": len(cleaned), "articles": cleaned}


"""
Tool Schemas (Groq tool-calling format)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city using a real weather API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name e.g. Karachi"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert money between currencies using live exchange rates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Search recent real news articles about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to search news for"}
                },
                "required": ["topic"]
            }
        }
    }
]


def run_tool(tool_name: str, tool_args: dict) -> dict:
    print(f"    Tool     : {tool_name}")
    print(f"    Arguments: {tool_args}")

    dispatch = {
        "get_weather": get_weather,
        "convert_currency": convert_currency,
        "get_news": get_news,
    }

    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}

    return fn(**tool_args)


"""
Groq retry wrapper - Groq's Llama models occasionally emit a malformed
tool call and throw a tool_use_failed error. Retry a couple times, then
fall back to a plain-text answer so the agent never crashes.
"""

def call_with_tool_retry(messages, tools, max_retries: int = 2):
    last_error = None

    for attempt in range(max_retries + 1):
        is_last_attempt = attempt == max_retries
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="none" if is_last_attempt else "auto"
            )
        except BadRequestError as e:
            last_error = e
            error_code = ""
            if hasattr(e, "body") and isinstance(e.body, dict):
                error_code = e.body.get("error", {}).get("code", "")

            if error_code == "tool_use_failed":
                print(f"    [retry {attempt + 1}/{max_retries}] Malformed tool call, retrying...")
                time.sleep(0.5 * (attempt + 1))
                continue
            raise

    raise last_error


def agent(user_input: str) -> str:
    print()
    print("User:", user_input)
    print()

    messages = [{"role": "user", "content": user_input}]

    response = call_with_tool_retry(messages, TOOLS)
    ai_message = response.choices[0].message

    if not ai_message.tool_calls:
        print("Agent:", ai_message.content)
        return ai_message.content

    tool_names = [tc.function.name for tc in ai_message.tool_calls]
    print(f"Tools selected: {tool_names}")
    print()

    messages.append(ai_message)

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments
        tool_args = json.loads(raw_args) if raw_args else {}

        tool_result = run_tool(tool_name, tool_args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_result)
        })

    print()


    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="none"
    )
    final_answer = final_response.choices[0].message.content
    print("Agent:", final_answer)
    return final_answer


"""
Tests
Phase 1: Each real API tool working correctly
Phase 2: Each tool's error handling (bad input, missing key, etc.)
"""

def run_tests():

    print()
    print("DAY 4 - EXTERNAL APIS AS TOOLS")
    print("Testing real weather / currency / news APIs + error handling")
    print()

    print("PHASE 1 - Real API tools working correctly")
    agent("What's the weather like in Karachi?")
    agent("Convert 500 USD to PKR")
    agent("Find news about artificial intelligence")

    print()
    print("PHASE 2 - Error handling checks")
    agent("What's the weather in Xyzzyplonk123?")        # invalid city
    agent("Convert 100 USD to FAKECOIN")                  # invalid currency
    agent("Find news about the economy")                  # will fail cleanly if NEWS_API_KEY missing


if __name__ == "__main__":
    run_tests()