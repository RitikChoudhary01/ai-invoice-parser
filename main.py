import os
import io
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import pypdf

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

# =========================================================
# TUNABLES FOR LATENCY
# =========================================================
MODEL_NAME = os.getenv(
        "INVOICE_MODEL",
        "deepseek/deepseek-v4-flash:free"
)
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "0"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "0"))

SYSTEM_PROMPT = """
You are an expert accounting AI.

Extract invoice information from the provided text.

Return ONLY valid JSON.

Required JSON structure:

{
    "vendor_name": "string",
    "invoice_number": "string",
    "date_issued": "YYYY-MM-DD",
    "total_amount": 0,
    "currency": "USD",
    "items": [
        {
            "description": "string",
            "amount": 0
        }
    ]
}

Rules:
- No markdown
- No explanations
- Only raw JSON
- Amounts must be numbers
- Missing values should be empty strings
""".strip()

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="AI Financial Reconciliation API",
    description="DeepSeek-powered invoice extraction engine",
    version="2.0.0"
)

# =========================================================
# OPENROUTER CLIENT
# =========================================================
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": os.getenv(
            "OPENROUTER_HTTP_REFERER",
            "http://localhost:8000"
        ),
        "X-Title": "AI Financial Reconciliation API",
    },
)

# =========================================================
# PYDANTIC MODELS
# =========================================================
class LineItem(BaseModel):
    description: str = Field(
        description="Description of product or service"
    )
    amount: float = Field(
        description="Cost of this item"
    )


class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: str
    date_issued: str
    total_amount: float
    currency: str
    items: list[LineItem]


# =========================================================
# ROOT ENDPOINT
# =========================================================
@app.get("/")
def root():
    return {
        "message": "DeepSeek Invoice Engine Running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health_check():
    return {
        "status": "active",
        "model": MODEL_NAME
    }


# =========================================================
# MAIN AI ENDPOINT
# =========================================================
@app.post(
    "/api/v1/process-document",
    response_model=InvoiceData
)
async def process_financial_document(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # VALIDATE FILE
    # -----------------------------------------------------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:

        # -------------------------------------------------
        # READ PDF
        # -------------------------------------------------
        pdf_bytes = await file.read()

        pdf_reader = pypdf.PdfReader(
            io.BytesIO(pdf_bytes)
        )

        # -------------------------------------------------
        # EXTRACT TEXT
        # -------------------------------------------------
        pages = pdf_reader.pages
        if MAX_PDF_PAGES > 0:
            pages = pages[:MAX_PDF_PAGES]

        chunks = []
        for page in pages:
            text = page.extract_text()
            if text:
                chunks.append(text)

        raw_text = "\n".join(chunks)

        # Optional cap to reduce tokens and speed up replies.
        if MAX_TEXT_CHARS > 0 and len(raw_text) > MAX_TEXT_CHARS:
            raw_text = raw_text[:MAX_TEXT_CHARS]

        # -------------------------------------------------
        # EMPTY PDF CHECK
        # -------------------------------------------------
        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="PDF is empty or unreadable."
            )

        # -------------------------------------------------
        # AI CALL (DEEPSEEK)
        # -------------------------------------------------
        completion = client.chat.completions.create(
                        model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                                        "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0
        )

        # -------------------------------------------------
        # GET AI RESPONSE
        # -------------------------------------------------
        ai_response = completion.choices[0].message.content

        # -------------------------------------------------
        # PARSE JSON
        # -------------------------------------------------
        parsed_json = json.loads(ai_response)

        # -------------------------------------------------
        # VALIDATE USING PYDANTIC
        # -------------------------------------------------
        validated_data = InvoiceData(**parsed_json)

        return validated_data

    except HTTPException:
        raise

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid JSON."
        )

    except Exception as e:
        print("CRITICAL ERROR:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )