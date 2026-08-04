"""
Error Recovery Agent

Builds an agent that does not give up when a tool fails.
Instead it tries a backup tool automatically.
Every attempt is logged so you can see exactly what happened.
If a tool fails due to rate limiting, it waits longer each time
before retrying (this is called exponential backoff).

How it works:
    1. Try the primary tool
    2. If rate limited  -> wait and retry (1s, 2s, 4s...)
    3. If still failing -> try the backup tool instead
    4. Log every attempt (success or failure)

Tools in this lab:
    - get_weather    : Open-Meteo (primary) + wttr.in (backup)
    - search_news    : NewsAPI (primary)    + local mock index (backup)
    - demo tools     : two fake broken tools to show the recovery logic


"""

import os
import time
import requests
from dotenv import load_dotenv


load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


""" 
Attempt Log
Every tool call gets recorded here.
We print the full log at the end to show what happened.
"""

ATTEMPT_LOG = []


def log(tool_name: str, status: str, detail: str = ""):

    entry = {
        "tool":   tool_name,
        "status": status,
        "detail": detail
    }
    ATTEMPT_LOG.append(entry)
    print(f"    LOG [{tool_name}] {status} - {detail}")


def print_full_log():
    print()
    print("FULL ATTEMPT LOG")
    print()
    for i, entry in enumerate(ATTEMPT_LOG, start=1):
        print(f"  {i}. tool={entry['tool']} | status={entry['status']} | detail={entry['detail']}")


"""
Weather Tool - Primary
Uses Open-Meteo. Free, no API key needed.
Step 1: get coordinates for the city
Step 2: get current temperature from those coordinates
"""

def get_weather_primary(city: str) -> dict:

    # Step 1: get coordinates
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=8
    ).json()

    if not geo_response.get("results"):
        return {"error": f"City '{city}' not found"}

    place     = geo_response["results"][0]
    latitude  = place["latitude"]
    longitude = place["longitude"]

    # Step 2: get temperature
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":  latitude,
            "longitude": longitude,
            "current":   "temperature_2m"
        },
        timeout=8
    ).json()

    temperature = weather_response["current"]["temperature_2m"]

    return {
        "city":          place["name"],
        "temperature_c": temperature
    }


"""
Weather Tool - Backup
Uses wttr.in. Different provider, also free, no key.
Only runs if the primary fails.
"""

def get_weather_backup(city: str) -> dict:

    response = requests.get(
        f"https://wttr.in/{city}",
        params={"format": "3"},
        timeout=8
    )

    if response.status_code != 200:
        return {"error": "Backup weather service also failed"}

    return {
        "city":    city,
        "summary": response.text.strip()
    }


""" 
News Tool - Primary
Uses NewsAPI.org. Needs a free API key.
If no key is set, returns an error immediately.
If rate limited, returns rate_limited=True so backoff kicks in.
"""

def search_news_primary(topic: str) -> dict:

    if not NEWS_API_KEY:
        return {"error": "NEWS_API_KEY not set in .env file"}

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q":        topic,
            "pageSize": 3,
            "language": "en",
            "apiKey":   NEWS_API_KEY
        },
        timeout=8
    )

    # Rate limited - signal to the backoff logic
    if response.status_code == 429:
        return {"error": "Rate limited by NewsAPI", "rate_limited": True}

    if response.status_code != 200:
        return {"error": f"NewsAPI failed with status {response.status_code}"}

    articles = response.json().get("articles", [])

    if not articles:
        return {"error": "No articles found"}

    return {
        "topic":  topic,
        "titles": [article["title"] for article in articles]
    }


""" 
News Tool - Backup
A small local keyword index. No internet needed.
Always works. Runs only if the primary news tool fails.
"""

BACKUP_NEWS_INDEX = {
    "ai":      "AI companies continue investing billions in large language model research.",
    "climate": "Global temperatures remain a top concern for governments worldwide.",
    "economy": "Markets show mixed signals amid ongoing trade and inflation discussions.",
    "tech":    "Technology sector sees strong growth driven by AI and cloud computing.",
}

def search_news_backup(topic: str) -> dict:

    topic_lower = topic.lower()

    for keyword, headline in BACKUP_NEWS_INDEX.items():
        if keyword in topic_lower:
            return {"topic": topic, "titles": [headline]}

    return {
        "topic":  topic,
        "titles": [f"No specific news for '{topic}' but backup tool is working correctly."]
    }


""" 
Fake Tools - Only used for demonstrations
These always fail so we can show the recovery logic
working reliably without depending on real API failures.
"""

#its used to demonstrate the exponential backoff logic. It always returns a rate limit error, so the backoff logic will kick in and retry a few times before giving up. The backup tool will then be called and succeed.
def always_rate_limited(topic: str) -> dict:
    return {"error": "Rate limited (simulated for demo)", "rate_limited": True}

#its used to demonstrate the fallback logic. It always returns an error, so the primary tool will fail and the backup tool will be called instead.
def always_broken(city: str) -> dict:
    return {"error": "This tool is permanently broken (simulated for demo)"}


""" 
Exponential Backoff
If a tool says it is rate limited, we wait and retry.
Wait times: 1s, 2s, 4s, 8s... (doubles each time)
If it keeps failing after max_retries, we give up.
"""

def call_with_backoff(tool_func, tool_name: str, args: tuple, max_retries: int = 3) -> dict:
    
    for attempt in range(max_retries):

        result = tool_func(*args)

        # Rate limited - wait and retry
        if isinstance(result, dict) and result.get("rate_limited"):
            wait_seconds = 2 ** attempt
            log(tool_name, "rate_limited", f"attempt {attempt + 1} of {max_retries}, waiting {wait_seconds}s")
            time.sleep(wait_seconds)
            continue

        # Any other error - do not retry, just return the error
        if isinstance(result, dict) and "error" in result:
            log(tool_name, "failed", result["error"])
            return result

        # Success
        log(tool_name, "success", "tool returned a valid result")
        return result

    # Gave up after all retries
    log(tool_name, "failed", f"gave up after {max_retries} rate limit retries")
    return {"error": f"{tool_name} failed after {max_retries} retries"}


"""
Run Tool With Fallback
This is the main error recovery logic.
Try primary first. If it fails, try backup.
Both use backoff internally for rate limit errors.
"""

def run_tool_with_fallback(tool_name: str, primary_func, backup_func, args: tuple) -> dict:

    print(f"Trying primary: {tool_name}")
    primary_result = call_with_backoff(primary_func, f"{tool_name} (primary)", args)

    # Primary worked
    if "error" not in primary_result:
        return primary_result

    # Primary failed - try backup
    print(f"Primary failed. Trying backup: {tool_name}")
    backup_result = call_with_backoff(backup_func, f"{tool_name} (backup)", args)

    return backup_result


""" 
Tests

Test 1: Real weather - primary should work fine
Test 2: Real news   - falls back if no API key set
Test 3: Simulated rate limiting - shows backoff
Test 4: Simulated broken tool  - shows fallback
"""

def run_tests():

    # Clear the log before each full run
    global ATTEMPT_LOG
    ATTEMPT_LOG = []

    # Test 1 - Real weather, primary should succeed
    print()
    print("TEST 1 - Weather (primary should work)")
    print()
    result = run_tool_with_fallback(
        "get_weather",
        get_weather_primary,
        get_weather_backup,
        ("Karachi",)
    )
    print("Result:", result)


    # Test 2 - Real news, will fall back if no API key
    print()
    print("TEST 2 - News search (falls back if no API key set)")
    print()
    result = run_tool_with_fallback(
        "search_news",
        search_news_primary,
        search_news_backup,
        ("climate",)
    )
    print("Result:", result)


    # Test 3 - Fake tool that is always rate limited
    # Shows exponential backoff working: waits 1s, 2s, 4s then gives up
    # Then backup kicks in and succeeds
    print()
    print("TEST 3 - Rate limit demo (shows exponential backoff)")
    print()
    result = run_tool_with_fallback(
        "rate_limit_demo",
        always_rate_limited,
        search_news_backup,
        ("ai",)
    )
    print("Result:", result)


    # Test 4 - Fake tool that always fails immediately
    # Shows fallback working without needing to wait for backoff
    print()
    print("TEST 4 - Broken tool demo (shows fallback to backup)")
    print()
    result = run_tool_with_fallback(
        "weather_fallback_demo",
        always_broken,
        get_weather_backup,
        ("Lahore",)
    )
    print("Result:", result)


    # Print the full log at the end
    print_full_log()


if __name__ == "__main__":
    run_tests()