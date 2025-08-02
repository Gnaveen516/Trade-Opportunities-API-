# main.py
from fastapi import FastAPI, Request, HTTPException, Depends, status, Header
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
import time
import secrets
import json
import asyncio

# Tool: Google Search for simulating data collection
# In a real application, you would replace this with actual web scraping libraries
# or dedicated market data APIs (e.g., news APIs, financial data APIs).
# For this demonstration, we'll use a mock search.
def Google_Search_mock(query: str):
    """
    Simulates a Google search for market data and news.
    In a real application, this would use a proper web search API.
    """
    mock_responses = {
        "pharmaceuticals": f"Recent news for pharmaceuticals in India: Strong growth in generic drugs, increased R&D investments, regulatory changes, and rising demand for healthcare. Key players like Sun Pharma, Dr. Reddy's Laboratories, and Cipla are expanding. Export opportunities in Africa and Southeast Asia. Government policies supporting 'Make in India' for pharma. Challenges include price controls and competition.",
        "technology": f"Recent news for technology in India: Booming IT services, significant startup funding, focus on AI, ML, and cybersecurity. Digital transformation driving demand. Companies like TCS, Infosys, and Wipro are leading. Fintech and EdTech sectors are experiencing rapid expansion. Talent acquisition and infrastructure development are key areas. Increased foreign investment in tech startups.",
        "agriculture": f"Recent news for agriculture in India: Monsoon season outlook, government subsidies for farmers, adoption of modern farming techniques, challenges with climate change and supply chain. Focus on crop diversification, food processing, and agritech startups. Export demand for spices, rice, and fresh produce. Initiatives like e-NAM are digitalizing markets.",
        "automotive": f"Recent news for automotive in India: Shift towards electric vehicles (EVs), increased production for domestic and export markets, supply chain disruptions from chip shortages. Strong demand for SUVs. Maruti Suzuki, Tata Motors, and Hyundai are dominant. Government incentives for EV manufacturing and adoption. Focus on reducing emissions and localizing production.",
        "finance": f"Recent news for finance in India: Digital payments boom, expansion of fintech services, rising credit demand, regulatory reforms in banking sector. Public sector banks undergoing reforms. Increased foreign direct investment in financial services. Challenges include NPAs and cybersecurity threats. UPI transactions continue to break records.",
    }
    return mock_responses.get(query.lower(), f"No specific current data found for {query}. General market trends suggest economic growth and digital adoption.")

# FastAPI Application Initialization
app = FastAPI(
    title="Trade Opportunities API",
    description="A FastAPI service that analyzes market data and provides trade opportunity insights for specific sectors in India using the Gemini API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# In-memory storage for user sessions and rate limits
# In a real application, this would be a persistent store like Redis or a database.
session_store: Dict[str, Dict[str, Any]] = {}
user_rate_limits: Dict[str, Dict[str, Any]] = {}

# Configuration for Rate Limiting
RATE_LIMIT_DURATION = 60  # seconds
RATE_LIMIT_REQUESTS = 5   # requests per duration

# Simple API Key Authentication (for demonstration purposes)
# In a real application, use a proper authentication mechanism (e.g., OAuth2, JWT).
API_KEYS = {
    "mysecretapikey123": {"user_id": "test_user_1", "name": "Developer A"},
    "anothersecretkeyabc": {"user_id": "analyst_beta", "name": "Analyst B"},
}

bearer_scheme = HTTPBearer()

async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Authenticates user based on a simple API key in the Authorization Bearer header.
    """
    token = credentials.credentials
    user_info = API_KEYS.get(token)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_info["user_id"]

async def rate_limit_dependency(user_id: str = Depends(get_current_user_id)):
    """
    Applies rate limiting based on user ID.
    """
    current_time = time.time()
    
    if user_id not in user_rate_limits:
        user_rate_limits[user_id] = {"last_reset": current_time, "request_count": 0}

    user_data = user_rate_limits[user_id]

    # Reset count if the duration has passed
    if current_time - user_data["last_reset"] > RATE_LIMIT_DURATION:
        user_data["last_reset"] = current_time
        user_data["request_count"] = 0
    
    # Check if limit exceeded
    if user_data["request_count"] >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {int(RATE_LIMIT_DURATION - (current_time - user_data['last_reset']))} seconds.",
        )
    
    user_data["request_count"] += 1
    return user_id

async def call_gemini_api(prompt: str) -> str:
    """
    Calls the Gemini API to generate content.
    Implements exponential backoff for retries.
    """
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    payload = {"contents": chat_history}
    api_key = "YOUR_GEMINI_API_KEY_HERE"  
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    retries = 0
    max_retries = 3
    base_delay = 1  # seconds

    while retries < max_retries:
        try:
            # Using httpx for async requests within FastAPI
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url,
                    headers={'Content-Type': 'application/json'},
                    json=payload,
                    timeout=30 # Add a timeout for the API call
                )
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            result = response.json()

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                # Handle cases where the response structure is unexpected
                raise ValueError(f"Unexpected Gemini API response structure: {result}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and retries < max_retries - 1:
                delay = base_delay * (2 ** retries)
                retries += 1
                await asyncio.sleep(delay)
                continue
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Gemini API HTTP Error: {e.response.text}"
            )
        except httpx.RequestError as e:
            if retries < max_retries - 1:
                delay = base_delay * (2 ** retries)
                retries += 1
                await asyncio.sleep(delay)
                continue
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Gemini API Request Error: {e}"
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error parsing Gemini API response: {e}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred with Gemini API: {e}"
            )
    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail="Gemini API request timed out after multiple retries."
    )


def generate_markdown_report(sector: str, market_data: str, ai_analysis: str) -> str:
    """
    Generates a structured markdown report from market data and AI analysis.
    """
    report_content = f"""# Market Analysis Report: {sector.title()} Sector in India

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S IST')}

---

## 📊 Current Market Data & News Summary

The following information was gathered from recent market data and news articles related to the {sector.title()} sector in India:

{market_data}

---

## 📈 AI-Powered Trade Opportunity Insights

Based on the latest market information, our AI model (Gemini) provides the following insights and potential trade opportunities for the {sector.title()} sector:

{ai_analysis}

---

## ✅ Key Takeaways & Recommendations

* **Overall Outlook:** [Summarize the overall trend based on AI analysis]
* **High-Potential Areas:** [List specific sub-sectors or business models identified by AI]
* **Risks/Challenges:** [Mention challenges identified by AI]
* **Consider for Investment/Action:** [Specific actionable advice if provided by AI, otherwise a general prompt]

---

**Disclaimer:** This report is generated by an AI model based on available data and should not be considered financial advice. Always conduct your own thorough research and consult with financial professionals before making any investment decisions.
"""
    return report_content

# Main API Endpoint
@app.get("/analyze/{sector}", response_class=PlainTextResponse)
async def analyze_sector(
    sector: str,
    user_id: str = Depends(rate_limit_dependency) # Apply rate limiting and authentication
):
    """
    Analyzes market data for a given sector in India and returns a structured markdown report.

    - **sector**: The name of the market sector (e.g., "pharmaceuticals", "technology", "agriculture").
    - **Returns**: A structured markdown report with current trade opportunities.
    """
    try:
        # Step 1: Accept sector name as input (handled by FastAPI path parameter)
        if not sector or not isinstance(sector, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sector name must be a non-empty string."
            )
        
        # Input Validation example: ensure sector is alphanumeric/safe
        if not sector.replace(" ", "").isalnum(): # Allow spaces, but check alphanumeric otherwise
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid sector name. Only alphanumeric characters and spaces are allowed."
            )

        # Step 2: Search for current market data/news for that sector
        # IMPORTANT: In a real app, this would be a robust web scraping or news API call.
        # For this example, we're using a mock search.
        market_data_prompt = f"Find recent market data, news headlines, and key trends for the {sector} sector in India, specifically looking for information that highlights trade opportunities or challenges. Summarize briefly."
        
        # Using the provided Google Search_mock as a placeholder for data collection
        retries = 0
        max_retries = 3
        data_collection_result = None
        while retries < max_retries:
            try:
                data_collection_result = Google_Search_mock(sector) # Using the mock function
                if data_collection_result:
                    break
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(2 ** retries) # Exponential backoff for mock data
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to collect market data after multiple retries: {e}"
                    )
        
        if not data_collection_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to collect market data for the specified sector."
            )

        # Step 3: Use Gemini API to analyze collected information
        analysis_prompt = f"""Analyze the following market data for the {sector} sector in India.
        Identify key trends, growth drivers, challenges, and specific trade or investment opportunities.
        Focus on actionable insights.
        Market Data:
        {data_collection_result}
        
        Please provide a concise analysis, highlighting trade opportunities, in plain text format suitable for a markdown report.
        """
        ai_analysis_text = await call_gemini_api(analysis_prompt)

        # Step 4: Generate a structured markdown report
        markdown_report = generate_markdown_report(sector, data_collection_result, ai_analysis_text)

        # Step 5: Apply rate limiting and security measures (handled by rate_limit_dependency)

        return PlainTextResponse(content=markdown_report, media_type="text/markdown")

    except HTTPException as e:
        # Re-raise HTTPExceptions as they contain appropriate status codes and details
        raise e
    except Exception as e:
        # Catch any other unexpected errors and return a 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# Example of how to run this file:
# 1. Save this code as `main.py`
# 2. Make sure you have FastAPI and Uvicorn installed:
#    pip install fastapi uvicorn httpx
# 3. Run the application from your terminal:
#    uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Then, you can access the API documentation at http://localhost:8000/docs
# And make requests like:
# curl -X GET "http://localhost:8000/analyze/pharmaceuticals" \
#      -H "Authorization: Bearer mysecretapikey123" \

#      -H "accept: text/markdown"
