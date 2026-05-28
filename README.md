# Invoice Parser

A lightweight FastAPI service that extracts structured invoice data from raw
plain-text using the **OpenRouter** OpenAI-compatible API and structured outputs.

## Requirements

- Python ≥ 3.10
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

```bash
# 1. Clone / enter the project directory
cd invoice-parser

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your OpenRouter API key
# Edit .env and set OPENROUTER_API_KEY=sk-or-your-key-here
```

## Running

```bash
# Preferred during development
fastapi dev main.py

# Or run directly with Uvicorn
uvicorn main:app --reload
```

The server starts at **http://localhost:8000**.

Interactive API docs: http://localhost:8000/docs  
OpenAPI spec: http://localhost:8000/openapi.json

## Usage

Upload a PDF invoice to the `/api/v1/process-document` endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/process-document \
  -F "file=@invoice.pdf"
```

**Response:**

```json
{
  "vendor_name": "GreenTech Solutions Inc.",
  "invoice_number": "INV-2024-0781",
  "date_issued": "2024-03-12",
  "total_amount": 4598.23,
  "currency": "USD",
  "line_items": [
    "Solar Panel 400W x10 @ $199.00 = $1,990.00",
    "Inverter X7-5K x2 @ $899.00 = $1,798.00",
    "Mounting Kit MK-12 x10 @ $45.00 = $450.00"
  ]
}
```

## Project Structure

```
.
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md            # This file
```
