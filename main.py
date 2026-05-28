import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import pypdf

# Load environment variables
load_dotenv()

# Initialize the backend API
app = FastAPI(
    title="AI Financial Reconciliation API",
    description="Automated pipeline to extract and structure unstructured invoice data.",
    version="1.0.0"
)

# Initialize OpenRouter Client
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
        "X-Title": "AI Financial Reconciliation API",
    },
)

# ---------------------------------------------------------
# THE SCHEMA: This forces the AI to output perfect data
# ---------------------------------------------------------
class LineItem(BaseModel):
    description: str = Field(description="Description of the service or product")
    amount: float = Field(description="The cost of this specific line item")

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Name of the company issuing the invoice")
    invoice_number: str = Field(description="The unique invoice ID")
    date_issued: str = Field(description="Date in YYYY-MM-DD format")
    total_amount: float = Field(description="The final total amount billed")
    currency: str = Field(description="Currency code like USD, GBP, or INR")
    items: list[LineItem] = Field(description="List of individual items billed")

# ---------------------------------------------------------
# THE CORE ENDPOINT: Where the magic happens
# ---------------------------------------------------------
@app.post("/api/v1/process-document", response_model=InvoiceData)
async def process_financial_document(file: UploadFile = File(...)):
    # 1. Security Check: Ensure it's a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        # 2. Ingest the file into memory (No saving to hard drive = faster)
        pdf_bytes = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        
        # 3. Extract text from all pages
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text() + "\n"
            
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="The PDF appears to be empty or unreadable.")

        # 4. The AI Processing Pipeline
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini", # Fast, cheap, and highly accurate for routing
            messages=[
                {
                    "role": "system", 
                    "content": "You are a senior accounting AI. Extract the required financial data from the raw text. If a value is missing, infer it logically or leave it blank, but maintain the strict JSON structure."
                },
                {
                    "role": "user", 
                    "content": f"Extract data from this raw invoice text:\n\n{raw_text}"
                }
            ],
            response_format=InvoiceData,
        )

        # 5. Return the structured data to the client or n8n webhook
        return completion.choices[0].message.parsed

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error. Check logs.")

# Health check endpoint for cloud monitoring
@app.get("/health")
def health_check():
    return {"status": "active", "engine": "AI Reconciliation V1"}