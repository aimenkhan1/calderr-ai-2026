# 🌅 Morning Briefing Agent (API Aggregator)

A CLI agent that pulls data from multiple public APIs **in parallel**,
combines the results, and uses Groq to write it all up as a clean,
readable morning briefing — saved as a timestamped Markdown report.

---

## 📸 What It Does

Give it a city and a news topic, and the agent will:
- Fetch today's date, latest news headlines, a daily quote, and current weather
- Call all 3 external APIs **at the same time** (parallel, not sequential)
- Handle each API's failures independently — one API failing never breaks the others
- Synthesize everything into a short, friendly written briefing using an LLM
- Save the final report as a timestamped `.md` file
- Ask for city/topic through a simple CLI prompt

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Backend | Groq (llama-3.3-70b-versatile) |
| Weather API | Open-Meteo (free, no key) |
| News API | NewsAPI.org (free tier, needs a key) |
| Quote API | ZenQuotes.io (free, no key) |
| Parallel Calls | Python `concurrent.futures.ThreadPoolExecutor` |
| Environment | python-dotenv |
| Language | Python 3.11+ |

---

## 📂 Project Structure
api_aggregator_agent/

│

├── main.py            # Single file application

├── requirements.txt   # Dependencies

├── README.md          # This file

└── briefing_20260704_223202.md   # Sample generated report

---

## 📰 Briefing Sections

| Section | Emoji | Source |
|---------|-------|--------|
| Date | 📅 | Local system time (no API) |
| News | 📰 | NewsAPI.org |
| Daily Quote | 💬 | ZenQuotes.io |
| Weather | 🌤 | Open-Meteo |

---

## 🏗 Architecture
User Input (city + topic)

↓

Scheduler fires 3 API calls in parallel (ThreadPoolExecutor)

↓

News API ‖ Quote API ‖ Weather API (all running at once)

↓

Data Aggregator (combines all results into one dict)

↓

Groq LLM Synthesizer (writes the readable briefing)

↓

Formatted Markdown Report (.md file saved to disk)

↓

Printed to terminal + saved with timestamp

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/aimenkhan1/calderr-ai-2026.git
cd calderr-ai-2026/projects/week2-api-aggregator/api_aggregator_agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
NEWS_API_KEY=your_newsapi_key_here
```
Get a free Groq key at console.groq.com
Get a free NewsAPI key at newsapi.org/register

(Weather and quote APIs need no key.)

### 4. Run the agent
```bash
python main.py
```

---

## ⌨️ CLI Prompts

| Prompt | Default if left blank |
|--------|------------------------|
| City for weather | Karachi |
| News topic | technology |

---

## 💬 Example Run

```
========================================
MORNING BRIEFING AGENT
========================================

Enter a city for the weather (press Enter for Karachi): Lahore
Enter a news topic (press Enter for technology): AI

Fetching news, quote, and weather data in parallel...
Data fetched: {...}

Asking the LLM to write the briefing...

Building the final report...

Report saved to: briefing_20260704_153042.md
```

**Sample output (briefing_20260704_153042.md):**
```
# Morning Briefing

📅 Date: Saturday, July 04, 2026

📰 News: AI companies continue investing in large-scale research
as the industry accelerates development.

💬 Quote: "The secret of getting ahead is getting started." — Mark Twain

🌤 Weather: Lahore is currently 34°C.
```

---

## 📝 Notes

- News, quote, and weather calls all run in parallel via
  `ThreadPoolExecutor`, so total wait time is close to the *slowest*
  single API, not the sum of all three
- Each tool has independent error handling — a missing `NEWS_API_KEY`
  or an invalid city never crashes the whole run, it just shows up as
  a clean error in that one section
- Every run creates a new timestamped `.md` file, so previous reports
  are never overwritten — useful for keeping multiple sample reports
- Free-tier NewsAPI accounts have a daily request limit; hitting it
  returns a clear error instead of a crash

---

## 👩‍💻 Built By

Aiman Nadeem Khan