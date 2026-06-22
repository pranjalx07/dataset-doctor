# Dataset Doctor Agent 🩺

An AI multi-agent system that analyzes datasets to generate comprehensive reports on quality, bias, imbalance, and ML readiness, complete with optional voice explanations.

## Project Structure

```
dataset-doctor-agent/
│
├── backend/
│   ├── main.py                      # FastAPI entry point
│   ├── config.py                   # environment variables, API keys
│   ├── requirements.txt            # Python dependencies
│   ├── .env                        # secrets (NOT pushed to GitHub)
│
│   ├── agents/
│   │   ├── data_quality_agent.py   # missing values, duplicates, stats
│   │   ├── bias_agent.py           # class imbalance detection
│   │   ├── label_agent.py          # label consistency checks
│   │   ├── report_agent.py         # final report generator
│   │   └── orchestrator.py         # runs all agents together
│
│   ├── services/
│   │   ├── gemini_service.py       # LLM calls (optional)
│   │   ├── elevenlabs_service.py   # TTS audio generation
│
│   ├── utils/
│   │   ├── dataset_loader.py       # CSV loading & validation
│   │   ├── preprocessing.py        # cleaning helpers
│   │   ├── metrics.py              # reusable calculations
│
│   ├── uploads/                    # uploaded datasets (local only)
│   ├── outputs/                    # generated reports + audio files
│
│   └── __init__.py
│
│
├── frontend/
│   ├── app.py                      # Streamlit UI (main frontend)
│   ├── components/
│   │   ├── uploader.py            # file upload UI
│   │   ├── report_viewer.py       # show analysis results
│   │   └── audio_player.py        # play ElevenLabs output
│
│   └── assets/
│       └── styles.css             # optional styling
│
│
├── docs/
│   ├── architecture.md            # system design explanation
│   ├── kaggle_writeup.md          # submission report
│   └── demo_script.md             # 5-min video script
│
│
├── tests/
│   ├── test_agents.py             # basic unit tests
│   └── test_api.py                # FastAPI endpoint tests
│
│
├── .gitignore
├── README.md
└── LICENSE
```

## Setup & Running Instructions

### Backend
1. Navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```

### Frontend
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install streamlit:
   ```bash
   pip install streamlit
   ```
3. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
