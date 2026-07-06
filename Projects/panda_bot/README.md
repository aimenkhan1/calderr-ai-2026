# 📊 Data Analysis Agent

An AI-powered agent that reads any CSV file, writes pandas code to answer your question, runs it safely, and shows you the result as a chart, table, and plain English explanation.

---

## What It Does

Ask any question about your data in plain English and the agent will:

- 🧠 **Understand** — reads your CSV schema and question
- 💻 **Code** — generates correct pandas code using Groq
- 🔒 **Execute** — runs the code in a sandboxed subprocess safely
- 📊 **Visualize** — generates an appropriate bar, line, or pie chart
- 📝 **Explain** — writes a plain English narrative of the result
- 📥 **Report** — saves a downloadable markdown report

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Backend | Groq (`openai/gpt-oss-120b`) |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Data Validation | Pydantic v2 |
| UI | Streamlit |
| Sandbox | Python subprocess + tempfile |
| Environment | python-dotenv |

---

## Project Structure

```
data_analysis_agent/
│
├── app.py
│   └── Main Streamlit UI. Entry point of the application.
│       Handles file upload, question input, progress display,
│       chart + table results, download button, and sidebar history.
│
├── models.py
│   └── Pydantic data models used across all files.
│       Every piece of data is validated through these before use.
│
│       Models:
│         CSVSchema       - structure of the uploaded CSV file
│                           (filename, rows, columns, dtypes, sample)
│         GeneratedCode   - pandas code returned by Groq
│                           (question, code, explanation)
│         ExecutionResult - result of running code in the sandbox
│                           (success, output, error, output_type)
│         AnalysisReport  - final combined result for one analysis
│                           (question, code, result, narrative, chart_path)
│
├── analyzer.py
│   └── Reads an uploaded CSV and extracts its schema.
│       Returns a CSVSchema with column names, data types,
│       row count, and a 3-row sample preview sent to Groq.
│
├── code_generator.py
│   └── Sends the CSV schema and user question to Groq.
│       Groq returns Python pandas code that answers the question.
│       Also generates a plain English narrative of the result.
│
├── executor.py
│   └── Runs AI-generated code in a safe isolated subprocess.
│       Sandbox features:
│         - 10 second timeout (kills infinite loops)
│         - Temp folder isolation (deleted after run)
│         - Errors captured (app never crashes)
│         - No network access from generated code
│
├── visualizer.py
│   └── Takes the executor output and generates a chart.
│       Supports bar, line, and pie charts.
│       Uses dark purple theme to match the app.
│       Chart type is determined by a comment Groq adds to the code.
│
├── report_builder.py
│   └── Combines question, code, result, chart, and narrative
│       into a downloadable markdown report.
│       Reports are saved with timestamps in the reports/ folder.
│
├── requirements.txt
│   └── All Python dependencies needed to run the project.
│
├── sample_data/
│   ├── generate_samples.py
│   │   └── Run once to generate all 5 sample CSV files below.
│   │       Uses numpy and pandas to create realistic fake data.
│   │
│   ├── sales_data.csv        - product sales by region and date
│   ├── employees.csv         - employee salaries, departments, ratings
│   ├── ecommerce.csv         - order prices, categories, return rates
│   ├── monthly_revenue.csv   - revenue, expenses, profit by month
│   └── students.csv          - student scores and grades by subject
│
└── README.md
    └── This file.
```

---

## Architecture

```
CSV Upload
    ↓
analyzer.py         reads columns, types, row count → CSVSchema
    ↓
Question Input      plain English question from user
    ↓
code_generator.py   Groq writes pandas code → GeneratedCode
    ↓
executor.py         runs code in sandbox → ExecutionResult
    ↓
visualizer.py       generates bar / line / pie chart
    ↓
code_generator.py   Groq explains result in plain English
    ↓
report_builder.py   saves markdown report with timestamp
    ↓
app.py              chart + table + explanation + download button
```

---

## Data Flow

```
File uploaded
    ↓
CSVSchema created        (filename, rows, columns, dtypes, sample)
    ↓
GeneratedCode created    (question, pandas code, explanation)
    ↓
ExecutionResult created  (success, output, error, output_type)
    ↓
AnalysisReport created   (all of the above combined)
    ↓
Saved as .md report
Shown in Streamlit UI
```

---

## Setup and Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/calderr-ai-2026.git
cd calderr-ai-2026/week2/data_analysis_agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free key at [console.groq.com](https://console.groq.com)

### 4. Generate sample datasets
```bash
cd sample_data
python generate_samples.py
cd ..
```

### 5. Run the app
```bash
streamlit run app.py
```

Opens at http://localhost:8501

### Live Demo 
https://panda-bot.streamlit.app/


---

## Example Questions

| Dataset | Question |
|---------|---------|
| sales_data.csv | Which product had the highest total sales? |
| sales_data.csv | Show sales by region as a bar chart |
| employees.csv | What is the average salary by department? |
| employees.csv | Which department has the highest average rating? |
| ecommerce.csv | Which category has the highest average price? |
| ecommerce.csv | What percentage of orders were returned? |
| monthly_revenue.csv | Show the revenue trend over months |
| monthly_revenue.csv | Which month had the highest profit in 2025? |
| students.csv | What is the average score by subject? |
| students.csv | How many students passed vs failed? |

---

## Safety — The Sandbox

All AI-generated code runs in a sandboxed subprocess:

- **Timeout** — code is killed after 10 seconds (no infinite loops)
- **Isolation** — runs in a temp folder, deleted after execution
- **No network** — subprocess cannot make API calls
- **Error capture** — crashes shown to user, app never breaks

---

## Built By
Aiman Nadeem Khan