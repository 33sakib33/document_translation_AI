import os
import re
import asyncio
import time
import io
import json
import aiohttp
import streamlit as st
from docx import Document
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor
from pdf_utils import convert_pdf_to_docx_adobe

# --- CONFIGURATION ---
# Load PDF services API keys from backend (optional)
try:
    PDF_SERVICES_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID") or st.secrets.get("PDF_SERVICES_CLIENT_ID")
    PDF_SERVICES_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET") or st.secrets.get("PDF_SERVICES_CLIENT_SECRET")
except Exception:
    # Fallback for when Streamlit secrets aren't available
    PDF_SERVICES_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID")
    PDF_SERVICES_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET")

MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 10  # Send 10 paragraphs together

# --- Prompt Template ---
BATCH_PROMPT_TEMPLATE = """
Here are {count} passages from a manuscript that we want to translate into {language} to make it easily readable for everyone. These are the original English versions of the passages:

Can you translate each passage into fluent and natural {language}? Please rewrite them and do not plagiarize. Only give the translated text for each passage. 
Make sure that the reading experience "flows", for example by not using the same words & sentence structures too often. Only translate, do not mention anything else. Please format the text properly without separating lines between the passages and exactly in the same structure so I can easily copy/paste it entirely into the manuscript of the book. You can remove numbers if that makes the reading experience better. You must translate/rewrite everything exactly and do not shorten it. Keep the quotes in their formatted way if applicable.

This is important for my career: Please try to translate everything sentence-by-sentence and do not make it shorter!
Important: Please Translate everything sentence-by-sentence in fluent {language} and do not make it shorter!

This is crucial: When you encounter general terminology such as "chapter," "the end,", "introduction" etc., or phrases that are widely used across books, do not vary or rephrase them—this is to ensure consistency. Always translate the text into the targetted language
Always use the most widely accepted or officially published book title, chapter name, or term if multiple synonyms are possible — e.g., use Meditations, not Reflections; Letters from a Stoic, not Stoic Letters. "Introduction", not "getting started"
Also, always choose the most commonly published or recognized title, name or term in the target language — e.g., De Gedaanteverwisseling, not Metamorfose in Dutch. "Arme Peter" ipv "poor Peter" or "Armzalige Peter"

IMPORTANT: Return your response as a valid JSON object with the following structure:
{{
  "translations": [
    {{
      "id": 1,
      "translation": "translated text for passage 1"
    }},
    {{
      "id": 2,
      "translation": "translated text for passage 2"
    }}
    // ... continue for all passages
  ]
}}

Original Passages:
{passages}
"""

# --- Helper Functions ---
def is_meaningful_text(text):
    cleaned = re.sub(r'[\W_]+', '', text)
    return bool(cleaned.strip())

def is_decorative_only(text):
    stripped = text.strip()
    return not stripped or re.fullmatch(r"[^\w\s]+", stripped) or re.fullmatch(r"[A-Z]", stripped)

def sanitize_response(text: str) -> str:
    """Remove any <think>...</think> tokens and trim the response."""
    if not text:
        return text
    text = re.sub(r'(?is)<think>.*?</think>', '', text)
    text = re.sub(r'(?i)</?think>', '', text)
    return text.strip()

class TokenTracker:
    """Real-time token usage tracking with dashboard updates."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking data."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.batch_history = []
        self.start_time = time.time()
    
    def add_batch_tokens(self, input_tokens, output_tokens, batch_size):
        """Add tokens from a batch request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        self.batch_history.append({
            'batch': self.total_requests,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'batch_size': batch_size,
            'timestamp': time.time()
        })
    
    def get_stats(self):
        """Get current statistics."""
        elapsed_time = time.time() - self.start_time
        total_tokens = self.total_input_tokens + self.total_output_tokens
        
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': total_tokens,
            'total_requests': self.total_requests,
            'elapsed_time': elapsed_time,
            'tokens_per_second': total_tokens / elapsed_time if elapsed_time > 0 else 0,
            'avg_tokens_per_request': total_tokens / self.total_requests if self.total_requests > 0 else 0,
            'batch_history': self.batch_history
        }

async def call_openrouter_batch_api(session, prompt, model, openrouter_api_key, log_list=None):
    """
    Call OpenRouter API for batch processing with token tracking.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourdomain.com",  # Optional but recommended
        "X-Title": "EasyTranslate"                # Optional but helpful for tracking
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}  # Request JSON response
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            if log_list is not None:
                log_entry = f"[BATCH API] Attempt {attempt + 1}/{MAX_RETRIES} - Model: {model}"
                log_list.append(log_entry)
            
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                
                result = data["choices"][0]["message"]["content"].strip()
                
                # Get token usage from response
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
                
                if log_list is not None:
                    log_list.append(f"[SUCCESS] Received response ({len(result)} chars)")
                    log_list.append(f"[TOKENS] Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                
                return {
                    'text': result,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens
                }
                
        except Exception as e:
            if log_list is not None:
                log_list.append(f"[ERROR] Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    
    if log_list is not None:
        log_list.append("[FAILED] All retry attempts exhausted")
    return None

def create_batch_from_paragraphs(paragraphs, language):
    """Create structured batch prompt from multiple paragraphs."""
    passages_text = ""
    for i, para in enumerate(paragraphs, 1):
        passages_text += f"\nPassage {i} (ID: {i}):\n\"\"\"\n{para}\n\"\"\"\n"
    
    return BATCH_PROMPT_TEMPLATE.format(
        count=len(paragraphs),
        language=language,
        passages=passages_text
    )

def parse_structured_response(response_text, expected_count, log_list=None):
    """Parse structured JSON response into individual translations."""
    try:
        # Clean response text (remove markdown code blocks if present)
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON
        parsed_response = json.loads(cleaned_text)
        
        if log_list:
            log_list.append(f"[JSON] Successfully parsed structured response")
        
        # Extract translations in order
        translations = []
        if 'translations' in parsed_response:
            # Sort by ID to ensure correct order
            translation_items = sorted(parsed_response['translations'], key=lambda x: x.get('id', 0))
            
            for item in translation_items:
                if 'translation' in item:
                    translations.append(item['translation'])
                    
            if log_list:
                log_list.append(f"[JSON] Extracted {len(translations)} translations from structured response")
        
        return translations
        
    except json.JSONDecodeError as e:
        if log_list:
            log_list.append(f"[JSON ERROR] Failed to parse JSON: {str(e)}")
            log_list.append(f"[FALLBACK] Attempting fallback parsing...")
        
        # Fallback: try to extract translations using regex or line splitting
        return fallback_parse_response(response_text, expected_count, log_list)
    
    except Exception as e:
        if log_list:
            log_list.append(f"[PARSE ERROR] {str(e)}")
        return fallback_parse_response(response_text, expected_count, log_list)

def fallback_parse_response(response_text, expected_count, log_list=None):
    """Fallback parsing when JSON parsing fails."""
    if log_list:
        log_list.append("[FALLBACK] Using line-based parsing")
    
    # Split by lines and filter empty ones
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    translations = []
    current_translation = ""
    
    for line in lines:
        # Skip lines that look like headers or JSON structure
        if (line.startswith('Passage') or line.startswith('Translation') or 
            line == '---' or line.startswith('{') or line.startswith('}') or
            '"id":' in line or '"translation":' in line):
            if current_translation:
                translations.append(current_translation.strip())
                current_translation = ""
            continue
        
        # Accumulate translation text
        if current_translation:
            current_translation += "\n" + line
        else:
            current_translation = line
    
    # Add the last translation
    if current_translation:
        translations.append(current_translation.strip())
    
    # If we still don't have enough, try double newline splitting
    if len(translations) < expected_count:
        alt_splits = response_text.split('\n\n')
        alt_translations = [t.strip() for t in alt_splits if t.strip() and not t.startswith('{')]
        if len(alt_translations) >= expected_count:
            translations = alt_translations[:expected_count]
    
    if log_list:
        log_list.append(f"[FALLBACK] Extracted {len(translations)} translations")
    
    return translations

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return document object and file info."""
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    
    if ext == ".pdf":
        # Convert PDF to DOCX using Adobe (still requires temp file for Adobe API)
        with NamedTemporaryFile(delete=False, suffix=ext) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input.flush()
            input_path = temp_input.name
        
        with st.spinner("🔄 Converting PDF to DOCX using Adobe..."):
            converted_docx = convert_pdf_to_docx_adobe(input_path)
            doc = Document(converted_docx)
            # Clean up temp files
            os.unlink(input_path)
            if os.path.exists(converted_docx):
                os.unlink(converted_docx)
    else:
        # Process DOCX directly from memory
        uploaded_file.seek(0)
        doc = Document(io.BytesIO(uploaded_file.read()))
    
    return doc, ext

async def translate_docx_batch_async(uploaded_file_or_path, language, model, openrouter_api_key, progress_callback=None, token_callback=None):
    """
    Translate a DOCX file using OpenRouter API with batch processing and IN-MEMORY processing.
    Returns document bytes instead of saving to disk.
    """
    if not openrouter_api_key:
        raise ValueError("OpenRouter API Key is required. Please provide your API key.")
    
    # Initialize token tracker
    tracker = TokenTracker()
    
    # Handle both file objects and file paths for backwards compatibility
    if hasattr(uploaded_file_or_path, 'read'):
        # It's an uploaded file object
        doc, file_ext = process_uploaded_file(uploaded_file_or_path)
    else:
        # It's a file path (backwards compatibility)
        doc = Document(uploaded_file_or_path)
        file_ext = ".docx"
    
    paragraphs = doc.paragraphs
    translated_content = []
    logs = []
    sample_prompts = []
    
    logs.append(f"[START] Batch translation started for language: {language}")
    logs.append(f"[INFO] Document has {len(paragraphs)} total paragraphs")
    logs.append(f"[INFO] Batch size: {BATCH_SIZE} paragraphs per request")
    logs.append(f"[INFO] Using OpenRouter model: {model}")
    logs.append(f"[INFO] Processing in memory - no files saved to disk")
    
    # Prepare paragraph batches
    paragraph_batches = []
    current_batch = []
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        original = para.text.strip()

        # Skip processing logic (same as original)
        if re.fullmatch(r"[A-Z]", original) and i + 1 < len(paragraphs) and paragraphs[i + 1].text.strip()[:1].isupper():
            p = para._element
            p.getparent().remove(p)
            paragraphs = doc.paragraphs
            logs.append(f"[SKIP] Removed single uppercase letter at para {i}")
            continue

        if not original or not is_meaningful_text(original) or is_decorative_only(original):
            i += 1
            continue

        word_count = len(original.split())
        is_heading = para.style.name.lower().startswith("heading") or para.alignment == 1

        if word_count <= 1:
            if not original.isupper() and not is_heading:
                i += 1
                continue

        # Add to current batch
        current_batch.append((i, para, original))
        
        # If batch is full or we're at the end, process it
        if len(current_batch) >= BATCH_SIZE or i == len(paragraphs) - 1:
            if current_batch:
                paragraph_batches.append(current_batch)
                current_batch = []
        
        i += 1

    total_batches = len(paragraph_batches)
    logs.append(f"[QUEUE] Prepared {total_batches} batches for translation")
    logs.append(f"[PROCESSING] Starting batch API requests...")
    
    # Create aiohttp session for API calls
    async with aiohttp.ClientSession() as session:
        # Process batches sequentially for now (can be parallelized later)
        for batch_idx, batch in enumerate(paragraph_batches):
            batch_paragraphs = [item[2] for item in batch]  # Extract text
            
            # Create batch prompt
            batch_prompt = create_batch_from_paragraphs(batch_paragraphs, language)
            
            # Store sample prompt
            if len(sample_prompts) < 3:
                sample_prompts.append({
                    "type": f"batch_{batch_idx + 1}",
                    "prompt": batch_prompt,
                    "original": f"{len(batch_paragraphs)} paragraphs in batch"
                })
            
            logs.append(f"[BATCH {batch_idx + 1}/{total_batches}] Processing {len(batch_paragraphs)} paragraphs...")
            
            # Call API
            batch_result = await call_openrouter_batch_api(session, batch_prompt, model, openrouter_api_key, logs)
            
            if batch_result:
                # Update token tracking
                tracker.add_batch_tokens(
                    batch_result['input_tokens'],
                    batch_result['output_tokens'],
                    len(batch_paragraphs)
                )
                
                # Update dashboard if callback provided
                if token_callback:
                    token_callback(tracker.get_stats())
                
                # Parse structured response
                batch_translations = parse_structured_response(batch_result['text'], len(batch_paragraphs), logs)
                
                logs.append(f"[BATCH {batch_idx + 1}] Received {len(batch_translations)} translations")
                
                # Validate we got the expected number of translations
                if len(batch_translations) != len(batch_paragraphs):
                    logs.append(f"[WARNING] Expected {len(batch_paragraphs)} translations, got {len(batch_translations)}")
                    # Pad with empty strings if we're short
                    while len(batch_translations) < len(batch_paragraphs):
                        batch_translations.append("[Translation missing]")
                    # Trim if we got too many
                    batch_translations = batch_translations[:len(batch_paragraphs)]
                
                # Apply translations to document
                for (para_idx, para, original), translation in zip(batch, batch_translations):
                    if translation and translation.strip():
                        translation = sanitize_response(translation)
                        translated_content.append(translation)
                        
                        # Update paragraph text in memory
                        for run in para.runs:
                            run.text = ""
                        if para.runs:
                            para.runs[0].text = translation
                        else:
                            para.add_run(translation)
                    else:
                        # Still append something to maintain consistency
                        fallback_text = f"[Translation failed for paragraph {para_idx}]"
                        translated_content.append(fallback_text)
            else:
                # Batch failed completely, add fallback content
                logs.append(f"[BATCH ERROR] Batch {batch_idx + 1} failed completely, adding fallback content")
                for para_idx, para, original in batch:
                    fallback_text = f"[Batch translation failed - Original: {original[:100]}...]"
                    translated_content.append(fallback_text)
            
            # Update progress
            if progress_callback:
                progress_callback((batch_idx + 1) / total_batches)
    
    # Save document to memory buffer instead of disk
    output_buffer = io.BytesIO()
    doc.save(output_buffer)
    output_buffer.seek(0)  # Reset buffer position to start
    
    final_stats = tracker.get_stats()
    
    logs.append(f"[SAVE] Document saved to memory buffer")
    logs.append(f"[TOKENS] Final usage - Input: {final_stats['total_input_tokens']}, Output: {final_stats['total_output_tokens']}, Total: {final_stats['total_tokens']}")
    logs.append(f"[DONE] Translation complete!")
    
    # Return document bytes instead of file path
    return output_buffer.getvalue(), translated_content, logs, sample_prompts, final_stats

# --- Streamlit UI ---
st.title("📘 Easy Translate V2: Batch Translation (OpenRouter)")
st.caption("🚀 Batch Processing • 🎯 Real-time Token Tracking • 📊 Performance Dashboard • 💾 In-Memory Processing")

# API Key input section
st.subheader("🔑 API Configuration")
openrouter_api_key = st.text_input(
    "Enter your OpenRouter API Key",
    type="password",
    help="Your API key is not stored and only used for this translation session. Get your key from https://openrouter.ai/keys",
    placeholder="Enter your OpenRouter API key here..."
)

if not openrouter_api_key:
    st.info("👆 Please enter your OpenRouter API key to continue. Your key is used locally and not stored.")
    st.markdown(
        """
        **How to get your API key:**
        1. Visit [OpenRouter](https://openrouter.ai/keys)
        2. Sign in with your account
        3. Create or copy your API key
        4. Paste it in the field above
        
        🔒 **Privacy**: Your API key is only used during this session and is never stored on our servers.
        """
    )

st.divider()

uploaded_file = st.file_uploader("Upload a .docx or .pdf manuscript", type=["pdf", "docx"])

# Model and language selection
col1, col2 = st.columns([2, 2])
with col1:
    target_language = st.selectbox("Select output language", ["Contemporary English", "Spanish", "German", "Dutch"])
with col2:
    openrouter_model = st.selectbox(
        "Select Model",
        [   "google/gemini-2.5-flash",
             "google/gemini-2.5-pro",
            "qwen/qwen3-235b-a22b:free",
            "anthropic/claude-sonnet-4.5",

        ],
        help="Choose a model from OpenRouter's catalog"
    )

# Info about processing
st.info("💾 **In-Memory Processing**: No files are saved to local storage - perfect for secure environments.")

if uploaded_file and openrouter_api_key:
    base_name = os.path.splitext(uploaded_file.name)[0]
    lang_suffix = target_language.lower().replace(" ", "_")
    final_filename = f"{base_name}_{lang_suffix}_v2.docx"

    if st.button("🚀 Start Batch Translation"):
        # Create placeholder containers for real-time updates
        progress_bar = st.progress(0)
        token_dashboard = st.empty()
        
        def update_progress(pct):
            progress_bar.progress(pct)

        def update_token_dashboard(stats):
            """Real-time token dashboard update."""
            
            with token_dashboard.container():
                st.subheader("📊 Real-Time Token Analytics")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "🎯 Input Tokens",
                        f"{stats['total_input_tokens']:,}",
                        help="Tokens sent to the model (including prompts)"
                    )
                with col2:
                    st.metric(
                        "📤 Output Tokens",
                        f"{stats['total_output_tokens']:,}",
                        help="Tokens received from the model"
                    )
                with col3:
                    st.metric(
                        "🔢 Total Tokens",
                        f"{stats['total_tokens']:,}",
                        help="Total tokens processed"
                    )
                with col4:
                    st.metric(
                        "⚡ Throughput",
                        f"{stats['tokens_per_second']:.1f}/s",
                        help="Tokens processed per second"
                    )
                
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📋 Batch Requests", f"{stats['total_requests']}")
                with col2:
                    st.metric("⏱️ Elapsed Time", f"{stats['elapsed_time']:.1f}s")
                with col3:
                    avg_tokens = stats['avg_tokens_per_request']
                    st.metric("📊 Avg Tokens/Batch", f"{avg_tokens:.0f}")
                
                # Batch history chart
                if stats.get('batch_history'):
                    st.markdown("### 📈 Batch Processing History")
                    batch_data = []
                    for batch in stats['batch_history']:
                        batch_data.append({
                            'Batch': batch['batch'],
                            'Input Tokens': batch['input_tokens'],
                            'Output Tokens': batch['output_tokens'],
                            'Total Tokens': batch['total_tokens'],
                            'Paragraphs': batch['batch_size']
                        })
                    
                    if batch_data:
                        st.dataframe(batch_data, use_container_width=True)

        with st.spinner("🔄 Processing batches with real-time analytics (in memory)..."):
            try:
                document_bytes, translated_content, logs, sample_prompts, final_stats = asyncio.run(
                    translate_docx_batch_async(
                        uploaded_file, 
                        target_language, 
                        openrouter_model,
                        openrouter_api_key,  # Pass API key from frontend
                        update_progress, 
                        update_token_dashboard
                    )
                )

                progress_bar.empty()
                st.success("✅ Batch translation completed! (Processed in memory)")
                
                # Final analytics summary
                st.subheader("🎯 Final Analytics Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Total Cost Est.", f"${final_stats['total_tokens'] * 0.000001:.4f}")
                with col2:
                    efficiency = final_stats['total_output_tokens'] / final_stats['total_input_tokens'] if final_stats['total_input_tokens'] > 0 else 0
                    st.metric("⚡ Efficiency Ratio", f"{efficiency:.2f}x")
                with col3:
                    batches_per_min = (final_stats['total_requests'] / final_stats['elapsed_time'] * 60) if final_stats['elapsed_time'] > 0 else 0
                    st.metric("🚀 Batches/Min", f"{batches_per_min:.1f}")
                with col4:
                    total_paragraphs = sum(b['batch_size'] for b in final_stats.get('batch_history', []))
                    st.metric("📄 Total Paragraphs", f"{total_paragraphs}")
                
                # Performance insights
                avg_batch_time = final_stats['elapsed_time'] / final_stats['total_requests'] if final_stats['total_requests'] > 0 else 0
                st.info(
                    f"🚀 **Performance Summary:** "
                    f"Processed {final_stats['total_requests']} batches in {final_stats['elapsed_time']:.1f}s "
                    f"(avg {avg_batch_time:.2f}s per batch) • "
                    f"Throughput: {final_stats['tokens_per_second']:.1f} tokens/sec"
                )
                
                st.divider()
                
                # Display logs section
                st.subheader("📋 Translation Logs")
                with st.expander("🔍 View Processing Logs", expanded=False):
                    log_text = "\n".join(logs)
                    st.code(log_text, language="log")
                
                # Display sample prompts
                if sample_prompts:
                    with st.expander("📝 Sample Batch Prompts", expanded=False):
                        for i, sample in enumerate(sample_prompts, 1):
                            st.markdown(f"**Batch #{i} - {sample['type'].replace('_', ' ').title()}**")
                            st.text_area(
                                f"Batch Info #{i}:",
                                value=sample['original'],
                                height=50,
                                key=f"batch_info_{i}"
                            )
                            st.text_area(
                                f"Full Batch Prompt #{i}:",
                                value=sample['prompt'],
                                height=300,
                                key=f"batch_prompt_{i}"
                            )
                            if i < len(sample_prompts):
                                st.divider()
                
                st.divider()
                
                # Display translated content
                st.subheader("📖 Translated Content Preview")
                full_text = "\n\n".join(translated_content)
                
                if full_text:
                    st.text_area(
                        "Translated text:", 
                        value=full_text, 
                        height=400, 
                        help="Preview of your translated document. You can scroll to read the full content.",
                        key="translated_preview"
                    )
                    
                    word_count = len(full_text.split())
                    st.info(f"📊 Translation contains {word_count:,} words across {len(translated_content)} paragraphs")
                else:
                    st.warning("No translated content to display.")
                
                st.divider()
                
                # Download section - using document bytes instead of file
                st.subheader("💾 Download Options")
                st.download_button(
                    "📥 Download Translated DOCX", 
                    document_bytes,  # Direct bytes instead of file reference
                    file_name=final_filename, 
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    help="Download the translated document as a Word file (processed entirely in memory)",
                    use_container_width=True
                )
                
                st.info("💡 **Tip**: The translation results will remain visible even after downloading. You can download the file multiple times if needed.")
            
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Translation failed: {str(e)}")
                st.exception(e)

elif uploaded_file and not openrouter_api_key:
    st.warning("⚠️ Please enter your OpenRouter API key above to start translation.")
