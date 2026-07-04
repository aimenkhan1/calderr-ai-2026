"""
API Aggregator Agent (Morning Briefing Agent)

Goal: pull data from multiple sources AT THE SAME TIME (in parallel,
not one after another), combine the results, and have an LLM write
it all up as a readable "morning briefing" report.

Sections in the briefing (in order):
    1. Day and Date       - no API needed, just today's date
    2. News                - NewsAPI.org (free tier, needs a key)
    3. Daily Quote          - ZenQuotes.io (free, no key needed)
    4. Weather of a city   - Open-Meteo (free, no key needed)


"""

import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


#this function gets the current day and date in a friendly format, e.g. "Monday, January 1, 2024"

def get_day_and_date():
    now = datetime.now()
    return {
        "day": now.strftime("%A"),
        "date": now.strftime("%B %d, %Y")
    }


#this function gets the latest news headlines for a given topic using NewsAPI.org (needs a free key)

def get_news(topic, max_results=3):

    try:
        if not NEWS_API_KEY:
            return {"error": "NEWS_API_KEY not set in .env file"}

        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic,
                "sortBy": "publishedAt",
                "pageSize": max_results,
                "language": "en",
                "apiKey": NEWS_API_KEY
            },
            timeout=8
        )

        # Check the actual status code before trusting the response body -
        # a bad key or rate limit still returns valid JSON, just with an
        # error message instead of articles, so we must check this first.
        if response.status_code == 401:
            return {"error": "NEWS_API_KEY is invalid or expired"}

        if response.status_code == 429:
            return {"error": "NewsAPI rate limit reached, try again later"}

        if response.status_code != 200:
            return {"error": f"News API failed with status {response.status_code}"}

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return {"error": "No news articles found for this topic"}

        headlines = [article["title"] for article in articles]

        return {"topic": topic, "headlines": headlines}

    except requests.exceptions.RequestException:
        return {"error": "News service unreachable"}
    except (KeyError, ValueError):
        return {"error": "News service returned an unexpected response"}


#this function gets a daily quote from ZenQuotes.io (free, no key needed)

def get_daily_quote():

    try:
        response = requests.get("https://zenquotes.io/api/random", timeout=8)

        if response.status_code != 200:
            return {"error": f"Quote service failed with status {response.status_code}"}

        data = response.json()

        if not data:
            return {"error": "No quote returned"}

        quote_data = data[0]

        return {
            "quote": quote_data.get("q"),
            "author": quote_data.get("a")
        }

    except requests.exceptions.RequestException:
        return {"error": "Quote service unreachable"}
    except (KeyError, IndexError, ValueError):
        return {"error": "Quote service returned an unexpected response"}


#this function gets the weather for a given city using Open-Meteo (free, no key needed)

def get_weather(city):

    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=8
        )

        if geo_response.status_code != 200:
            return {"error": f"Geocoding failed with status {geo_response.status_code}"}

        geo = geo_response.json()

        if not geo.get("results"):
            return {"error": f"City '{city}' not found"}

        place = geo["results"][0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m"
            },
            timeout=8
        )

        if weather_response.status_code != 200:
            return {"error": f"Forecast request failed with status {weather_response.status_code}"}

        weather = weather_response.json()

        return {
            "city": place["name"],
            "temperature_c": weather["current"]["temperature_2m"]
        }

    except requests.exceptions.RequestException:
        return {"error": "Weather service unreachable"}
    except (KeyError, ValueError):
        return {"error": "Weather service returned an unexpected response"}


""" 
Scheduler - fires the 3 REAL API calls at the same time
Day/date needs no API call, so it's just computed directly.
News, quote, and weather all call external websites, so those
three run in parallel using ThreadPoolExecutor - this means the
total wait is close to the SLOWEST single call, not the sum of
all three added together.
"""

def fetch_all_data(city, news_topic):

    date_data = get_day_and_date()  # here no API needed, just local time

    with ThreadPoolExecutor(max_workers=3) as executor:
        news_future = executor.submit(get_news, news_topic)
        quote_future = executor.submit(get_daily_quote)
        weather_future = executor.submit(get_weather, city)

        news_data = news_future.result()
        quote_data = quote_future.result()
        weather_data = weather_future.result()

    return {
        "date": date_data,
        "news": news_data,
        "quote": quote_data,
        "weather": weather_data
    }


# this function synthesizes the final briefing text using the aggregated data and an LLM

def synthesize_briefing(data):

    prompt = f"""
You are writing a short morning briefing for a busy person.
Use ONLY the data below. If any section has an "error" field,
briefly mention that this data was not available - do not make up
headlines, quotes, or numbers to fill the gap.

Date: {data['date']}
News: {data['news']}
Daily Quote: {data['quote']}
Weather: {data['weather']}

Write 4 short sections in this exact order: Date, News, Quote, Weather.
Keep the whole thing under 150 words. Friendly, clear tone.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


#this function builts the final markdown report using the synthesized briefing and raw data

def build_markdown_report(briefing_text, data):

    report = f"""# Morning Briefing

{briefing_text}

---

## Raw Data (for reference)

**Date:** {data['date']}

**News:** {data['news']}

**Quote:** {data['quote']}

**Weather:** {data['weather']}
"""
    return report



# Main Pipeline: Scheduler -> Aggregator -> Synthesizer -> Report


def run_morning_briefing(city="Karachi", news_topic="technology"):

    print("Fetching news, quote, and weather data in parallel...")
    data = fetch_all_data(city, news_topic)
    print("Data fetched:", data)

    print()
    print("Asking the LLM to write the briefing...")
    briefing_text = synthesize_briefing(data)

    print()
    print("Building the final report...")
    report = build_markdown_report(briefing_text, data)

    filename = f"briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print(f"Report saved to: {filename}")
    print()
    print(report)

    return report


if __name__ == "__main__":
    print()
    print("MORNING BRIEFING AGENT")
    print()
    print()

    city = ""
    while len(city) < 3:
        city = input("Enter a city for the weather: ").strip()
        if len(city) < 3:
            print("Please enter a correct, valid city name.")

    news_topic = ""
    while len(news_topic) < 3:
        news_topic = input("Enter a news topic: ").strip()
        if len(news_topic) < 3:
            print("Please enter a correct, valid news topic.")

    print()
    run_morning_briefing(city=city, news_topic=news_topic)