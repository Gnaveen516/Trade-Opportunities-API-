# ✨ Trade Opportunities API ✨

![Trade Opportunities API](https://wallpaperaccess.com/full/6227484.jpg)

A high-performance FastAPI service that delivers AI-powered market analysis and trade opportunity insights for specific sectors in India. This project is designed to be a rapid-deployment tool for market analysts, investors, and business strategists.

-----

## 🚀 Key Features
**📈 AI-Powered Insights: Utilizes the Google Gemini API to perform in-depth analysis of market data.**

**🌐 Dynamic Data Collection: Simulates real-time market news and trends for the specified sector.**

**📝 Structured Markdown Reports: Generates professional, human-readable reports that can be easily saved as .md files.**

**🔒 Robust Security: Implements API key authentication and rate limiting to prevent abuse and ensure fair usage.**

**⚡️ High Performance: Built on FastAPI's asynchronous framework for speed and efficiency.**

**📚 Self-Documenting: Comes with automatic interactive API documentation (Swagger UI/ReDoc).**

**💾 In-Memory Storage: Keeps the service lightweight and stateless, with no external database dependencies.**

## ⚙️ Setup & Installation
**1. Prerequisites
Before you begin, ensure you have the following installed:**

- Python 3.8+

- pip (Python package installer)

**2. Install Dependencies
It is highly recommended to use a virtual environment to manage dependencies for this project.**

### Create a virtual environment
``` sh
python -m venv venv
```
### Activate the virtual environment
### On macOS and Linux:
``` sh
source venv/bin/activate
```
### On Windows:
``` sh
venv\Scripts\activate
```
### Install the required packages
``` sh
pip install fastapi uvicorn httpx
```

**3. Obtain a Google Gemini API Key**

- The application requires a Gemini API key to perform market analysis.

- Go to the Google AI Studio: https://aistudio.google.com/

- Create or select a project.

- Navigate to "Get API key" and generate a new API key.

- Open the main.py file and locate the call_gemini_api function.

- Replace "YOUR_GEMINI_API_KEY_HERE" with your actual key:

- api_key = "YOUR_GEMINI_API_KEY_HERE"

**4. Run the Application**

- Execute the FastAPI application using Uvicorn. The --reload flag is great for development as it automatically restarts the server on code changes.

``` sh
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- The server will now be running at http://localhost:8000.

## 📖 API Usage
API Documentation
The project includes an interactive API documentation page powered by Swagger UI.

**👉 View Docs: http://localhost:8000/docs**

You can use this interface to test the endpoint directly.

- Making a Request
To get a market analysis report, send a GET request to the /analyze/{sector} endpoint. You must include a valid API key in the Authorization header.

**API Keys
Two example API keys are pre-configured for demonstration:**

- mysecretapikey123 (for user test_user_1)

- anothersecretkeyabc (for user analyst_beta)

**Example Request using curl**
``` sh
curl -X 'GET' \
  'http://localhost:8000/analyze/pharmaceuticals' \
  -H 'accept: text/markdown' \
  -H 'Authorization: Bearer mysecretapikey123'
```

**Note: The response will be a structured markdown report that you can save directly to a file (e.g., report.md).**

## 🔒 Security & Rate Limiting
**The API employs a simple API key authentication and rate-limiting system.**

- **Rate Limit:** 5 requests per minute (60 seconds) per authenticated user.

- **Authentication:** All requests must include a valid API key in the Authorization header.

#### **If you exceed the rate limit or provide an invalid key, you will receive an appropriate HTTP error response (429 Too Many Requests or 401 Unauthorized).**

## 🎨 Future Enhancements
**This project can be expanded upon with the following improvements:**

- **Real-time Data:** Integrate with a live news or financial data API (e.g., NewsAPI, Alpha Vantage) for a more dynamic analysis.

- **Persistent Storage:** Implement a database (like Redis or PostgreSQL) for long-term rate limiting, session management, and caching.

- **Advanced Security:** Upgrade authentication to a more robust system like OAuth2 or JWT with proper token management.

- **New Endpoints:** Add endpoints to allow users to save reports, view historical analyses, or get insights on custom data.

## 📝 Disclaimer
This API and its reports are generated by an AI model and should not be considered financial advice. Always conduct your own thorough research and consult with a professional before making any investment or trade decisions. The data used for analysis is simulated for this project's demonstration.

## 📄 License
This project is open-source and available under the MIT License.
