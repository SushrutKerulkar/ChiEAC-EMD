# Explain My Data

Upload any dataset and get clear, plain-English explanations with interactive visual summaries — powered by Groq.

**Live app:** [explainmydata.streamlit.app](https://explainmydata.streamlit.app/)

---

## What It Does

Drop a CSV, Excel, JSON, or Parquet file and instantly get:

- **AI-generated explanation** of your dataset — patterns, distributions, data quality issues, and key insights
- **Interactive charts** auto-generated based on what's in your data
- **Per-chart AI analysis** — click "Explain" on any chart for a plain-English breakdown
- **Free-form Q&A** — ask anything about your data in a chat interface
- **Automatic S3 backup** — every uploaded file is stored in the `csv/` folder of an S3 bucket

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend & UI | [Streamlit](https://streamlit.io/) |
| AI / LLM | [Groq API](https://groq.com/) — LLaMA 3.3 70B (`llama-3.3-70b-versatile`) |
| Charts | [Plotly](https://plotly.com/python/) (interactive, dark-themed) |
| Data processing | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| File parsing | Pandas + [OpenPyXL](https://openpyxl.readthedocs.io/) (Excel), [PyArrow](https://arrow.apache.org/docs/python/) (Parquet) |
| Cloud storage | [AWS S3](https://aws.amazon.com/s3/) via [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) |
| Config | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## Project Structure

```
explain_my_data/
├── app.py           # Streamlit frontend
│                    #   - Page layout and dark-mode styling
│                    #   - File upload widget (CSV, XLSX, XLS, JSON, Parquet)
│                    #   - S3 upload on file drop (stored under csv/<filename>)
│                    #   - Three tabs: Overview, Visual Analysis, Q&A
│
├── analyzer.py      # Data analysis and visualisation engine
│                    #   - load_dataset()       — reads any supported file format
│                    #   - get_column_types()   — classifies columns as numeric /
│                    #                            categorical / datetime
│                    #   - plot_overview()      — missing values bar, column type
│                    #                            pie, quick stats table
│                    #   - plot_distributions() — histograms with mean/median lines
│                    #   - plot_boxplots()      — outlier detection box plots
│                    #   - plot_correlation()   — lower-triangle correlation heatmap
│                    #   - plot_categoricals()  — horizontal bar charts per category
│                    #   - plot_timeseries()    — multi-panel time series
│                    #   - plot_pairplot()      — scatter matrix for numeric cols
│                    #   - plot_missing_pattern() — heatmap of missing value locations
│                    #   - generate_all_charts() — runs all plot functions and
│                    #                             returns only the ones that apply
│                    #   - build_data_summary() — builds a structured text summary
│                    #                            of the dataset for the LLM
│
├── explainer.py     # Groq API integration
│                    #   - explain_dataset()  — ≤8-bullet dataset summary
│                    #   - explain_chart()    — ≤4-bullet chart analysis
│                    #   - answer_question()  — free-form Q&A about the dataset
│                    #   Model: llama-3.3-70b-versatile
│
└── requirements.txt # Python dependencies
```

---

## Supported File Formats

| Format | Extensions |
|---|---|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| JSON | `.json` |
| Parquet | `.parquet` |
