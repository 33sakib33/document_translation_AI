# 📘 Easy Translate: Manuscript Translator

A powerful Streamlit application for translating manuscripts using AI. Supports OpenRouter, Google Gemini API, and local Ollama models.

## 🚀 Features

- ✅ Translate DOCX and PDF documents
- ✅ Multiple language support (Contemporary English, Spanish, German, Dutch)
- ✅ Real-time translation progress tracking
- ✅ In-app preview of translated content
- ✅ Detailed logging and prompt inspection
- ✅ Parallel API requests for faster translation
- ✅ Adobe PDF Services integration for PDF conversion
- ✅ **NEW:** Local translation with Ollama (no API keys needed!)

## 📦 Three Versions Available

### 1. **app.py** - OpenRouter Version

Uses OpenRouter API to access various AI models (currently configured for Gemini via OpenRouter)

**Required API Keys:**

- `OPENROUTER_API_KEY`
- `PDF_SERVICES_CLIENT_ID`
- `PDF_SERVICES_CLIENT_SECRET`

### 2. **app_gemini.py** - Direct Gemini API Version

Uses Google's official Gemini API directly (faster and more direct)

**Required API Keys:**

- `GEMINI_API_KEY`
- `PDF_SERVICES_CLIENT_ID`
- `PDF_SERVICES_CLIENT_SECRET`

### 3. **app_ollama.py** - Local Ollama Version 🆕

Runs completely locally using Ollama - no internet or API keys needed!

**Requirements:**

- Ollama installed locally ([Download here](https://ollama.ai))
- At least one model pulled (e.g., `ollama pull llama3.2`)
- `PDF_SERVICES_CLIENT_ID` and `PDF_SERVICES_CLIENT_SECRET` (only for PDF conversion)

## 🛠️ Installation

1. **Clone or download this repository**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up API keys** (choose one method):

   **Option A: Environment Variables**

   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   export PDF_SERVICES_CLIENT_ID="your-adobe-client-id"
   export PDF_SERVICES_CLIENT_SECRET="your-adobe-client-secret"
   ```

   **Option B: Streamlit Secrets**
   Create `.streamlit/secrets.toml`:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   OPENROUTER_API_KEY = "your-openrouter-api-key"
   PDF_SERVICES_CLIENT_ID = "your-adobe-client-id"
   PDF_SERVICES_CLIENT_SECRET = "your-adobe-client-secret"
   ```

## 🎮 Usage

### Run OpenRouter Version:

```bash
streamlit run app.py
```

### Run Gemini Direct Version:

```bash
streamlit run app_gemini.py
```

### Run Ollama Local Version:

```bash
# First, make sure Ollama is installed and running
ollama serve  # Usually auto-starts on macOS/Linux

# Pull a model if you haven't already
ollama pull llama3.2

# Run the app
streamlit run app_ollama.py
```

## 📖 How to Use

1. **Upload Document**: Upload a `.docx` or `.pdf` file
2. **Select Language**: Choose your target translation language
3. **Click Translate**: Hit the "🚀 Translate Now" button
4. **View Results**:
   - Check the processing logs
   - Inspect sample prompts sent to the API
   - Preview the translated content
   - Download the translated DOCX file

## 🔧 Configuration

### Adjustable Settings

In both `app.py` and `app_gemini.py`, you can modify:

```python
MAX_RETRIES = 3                    # Number of retry attempts for failed API calls
RETRY_DELAY = 5                    # Delay between retries (seconds)
MAX_CONCURRENT_REQUESTS = 5        # Number of parallel API requests
```

### Model Selection

**app.py (OpenRouter):**

```python
"model": "google/gemini-2.5-flash"  # Change to any OpenRouter model
```

**app_gemini.py (Direct Gemini):**

```python
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Change to gemini-2.5-flash, etc.
```

## 📊 Features in Detail

### Translation Logs 📋

- Track the entire translation process
- Monitor API call success/failure rates
- View processing steps and timing

### Sample Prompts 📝

- Inspect the first 3 prompts sent to the API
- Verify language instructions
- Debug translation issues

### Content Preview 📖

- View translated content before downloading
- Word and paragraph count statistics
- Scrollable text area for easy reading

### Download Options 💾

- Download translated DOCX file
- Filename automatically includes target language

## 🔑 Getting API Keys

### Google Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Use it as `GEMINI_API_KEY`

### OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up and create an API key
3. Use it as `OPENROUTER_API_KEY`

### Adobe PDF Services

1. Visit [Adobe Developer Console](https://developer.adobe.com/console)
2. Create a new project for PDF Services
3. Get your Client ID and Client Secret

## 🤔 Which Version to Use?

| Feature             | app.py (OpenRouter)     | app_gemini.py (Direct) | app_ollama.py (Local) |
| ------------------- | ----------------------- | ---------------------- | --------------------- |
| **Speed**           | Moderate                | Faster                 | Varies by hardware    |
| **Cost**            | OpenRouter pricing      | Google pricing         | **Free!** 🎉          |
| **Privacy**         | Cloud-based             | Cloud-based            | **100% Local** 🔒     |
| **Internet**        | Required                | Required               | **Not needed!** 🏠    |
| **Model Access**    | Multiple models         | Gemini only            | Any Ollama model      |
| **Setup**           | Need OpenRouter account | Need Google AI account | Install Ollama        |
| **Recommended For** | Testing multiple models | Production with Gemini | Privacy/Offline use   |

## 🏠 Why Use Ollama (app_ollama.py)?

- ✅ **No API Keys Required** - No signup, no billing, no limits
- ✅ **Complete Privacy** - Your documents never leave your machine
- ✅ **Work Offline** - No internet connection needed
- ✅ **No Per-Request Costs** - Translate unlimited documents
- ✅ **Multiple Models** - Choose from llama3.2, gemma2, mistral, qwen2.5, and more
- ✅ **Fully Open Source** - Complete transparency

### Popular Ollama Models for Translation:

| Model      | Size | Best For                       |
| ---------- | ---- | ------------------------------ |
| `llama3.2` | 3B   | Fast, balanced translations    |
| `llama3.1` | 8B+  | High-quality, detailed work    |
| `gemma2`   | 2-9B | Google's efficient model       |
| `mistral`  | 7B   | Quick, accurate translations   |
| `qwen2.5`  | 7B+  | Excellent multilingual support |
| `phi3`     | 3.8B | Compact, fast Microsoft model  |

## 📝 Notes

- PDF conversion requires Adobe PDF Services credentials
- Large documents may take several minutes to translate
- The app displays real-time progress during translation
- All API keys can be set via environment variables or Streamlit secrets

## 🐛 Troubleshooting

**"GEMINI_API_KEY not found"**

- Make sure your API key is set as an environment variable or in `.streamlit/secrets.toml`

**"Adobe conversion failed"**

- Verify your Adobe PDF Services credentials are correct
- Check if your Adobe account has sufficient credits

**Translation is slow**

- Increase `MAX_CONCURRENT_REQUESTS` (but be mindful of API rate limits)
- Use `gemini-2.0-flash-exp` for faster responses
- For Ollama: Use smaller models like `llama3.2` or `phi3`

**Ollama: "Connection refused" or "Model not found"**

- Make sure Ollama is running: `ollama serve`
- Check installed models: `ollama list`
- Pull the model: `ollama pull llama3.2`
- Verify Ollama is accessible: `curl http://localhost:11434`

**Ollama: Translation is very slow**

- Use smaller models (3B-7B parameters work best)
- Reduce `MAX_CONCURRENT_REQUESTS` to 2-3 for local inference
- Consider upgrading to a GPU-enabled machine for faster inference

## 📄 License

This project is provided as-is for manuscript translation purposes.

## 🙏 Credits

- Powered by Google Gemini / OpenRouter / Ollama
- PDF conversion by Adobe PDF Services
- UI built with Streamlit
- Local LLM inference by [Ollama](https://ollama.ai)
