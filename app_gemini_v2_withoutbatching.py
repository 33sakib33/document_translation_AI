import os
import re
import asyncio
import time
import io
from google.genai import types
import streamlit as st
from docx import Document
from tempfile import NamedTemporaryFile
from google import genai
from concurrent.futures import ThreadPoolExecutor
from pdf_utils import convert_pdf_to_docx_adobe

# --- CONFIGURATION ---
# Load API keys: prefer environment variables, fall back to Streamlit secrets
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    PDF_SERVICES_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID") or st.secrets.get("PDF_SERVICES_CLIENT_ID")
    PDF_SERVICES_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET") or st.secrets.get("PDF_SERVICES_CLIENT_SECRET")
except Exception:
    # Fallback for when Streamlit secrets aren't available
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PDF_SERVICES_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID")
    PDF_SERVICES_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET")

# Warn if API key is missing
if not GEMINI_API_KEY:
    st.warning("⚠️ GEMINI_API_KEY not found. Please set it as an environment variable or in Streamlit secrets.")

MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_CONCURRENT_REQUESTS = 5  # Process multiple paragraphs in parallel

# --- Prompt Template for Single Paragraph ---
SINGLE_PROMPT_TEMPLATE = """
Here is a passage from a manuscript that we want to translate into {language} to make it easily readable for everyone. This is the original English version:

\"\"\"{text}\"\"\"

Can you translate this passage into fluent and natural {language}? Please rewrite it and do not plagiarize. Only give the translated text. 
Make sure that the reading experience "flows", for example by not using the same words & sentence structures too often. Only translate, do not mention anything else. Please format the text properly in exactly the same structure so I can easily copy/paste it entirely into the manuscript of the book. You can remove numbers if that makes the reading experience better. You must translate/rewrite everything exactly and do not shorten it. Keep the quotes in their formatted way if applicable.

This is important for my career: Please try to translate everything sentence-by-sentence and do not make it shorter!
Important: Please Translate everything sentence-by-sentence in fluent {language} and do not make it shorter!

This is crucial: When you encounter general terminology such as "chapter," "the end,", "introduction" etc., or phrases that are widely used across books, do not vary or rephrase them—this is to ensure consistency. Always translate the text into the targetted language
Always use the most widely accepted or officially published book title, chapter name, or term if multiple synonyms are possible — e.g., use Meditations, not Reflections; Letters from a Stoic, not Stoic Letters. "Introduction", not "getting started"
Also, always choose the most commonly published or recognized title, name or term in the target language — e.g., De Gedaanteverwisseling, not Metamorfose in Dutch. "Arme Peter" ipv "poor Peter" or "Armzalige Peter"

Only return the translated text, nothing else.
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
        self.request_history = []
        self.start_time = time.time()
    
    def add_request_tokens(self, input_tokens, output_tokens):
        """Add tokens from a single request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        self.request_history.append({
            'request': self.total_requests,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
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
            'request_history': self.request_history
        }

def call_gemini_api(client, prompt, model, log_list=None):
    """
    Call Gemini API for single paragraph translation with token tracking.
    """
    for attempt in range(MAX_RETRIES):
        try:
            if log_list is not None:
                log_entry = f"[API] Attempt {attempt + 1}/{MAX_RETRIES} - Model: {model}"
                log_list.append(log_entry)
            
            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            
            result = response.text.strip()
            
            # Get token usage from response
            usage = response.usage_metadata
            actual_input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            total_tokens = usage.total_token_count
            
            if log_list is not None:
                log_list.append(f"[SUCCESS] Received response ({len(result)} chars)")
                log_list.append(f"[TOKENS] Input: {actual_input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            return {
                'text': result,
                'input_tokens': actual_input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens
            }
            
        except Exception as e:
            if log_list is not None:
                log_list.append(f"[ERROR] Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    if log_list is not None:
        log_list.append("[FAILED] All retry attempts exhausted")
    return None

async def call_gemini_async(executor, client, prompt, model, log_list=None):
    """Async wrapper for single API call."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, call_gemini_api, client, prompt, model, log_list)
    return result

def create_single_prompt(text, language):
    """Create prompt for single paragraph translation."""
    return SINGLE_PROMPT_TEMPLATE.format(
        text=text,
        language=language
    )

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

async def translate_docx_paragraph_by_paragraph(uploaded_file_or_path, language, model, progress_callback=None, token_callback=None):
    """
    Translate a DOCX file using Gemini API with PARAGRAPH-BY-PARAGRAPH processing (no batching).
    Returns document bytes instead of saving to disk.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please configure your API key.")
    
    # Initialize Gemini client and token tracker
    client = genai.Client(api_key=GEMINI_API_KEY)
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
    
    logs.append(f"[START] Paragraph-by-paragraph translation started for language: {language}")
    logs.append(f"[INFO] Document has {len(paragraphs)} total paragraphs")
    logs.append(f"[INFO] Processing mode: Individual paragraph requests (no batching)")
    logs.append(f"[INFO] Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    logs.append(f"[INFO] Using Gemini model: {model}")
    logs.append(f"[INFO] Processing in memory - no files saved to disk")
    
    # Prepare individual paragraph jobs
    jobs = []
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        original = para.text.strip()

        # Skip processing logic (same as batch version)
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

        # Create prompt for this single paragraph
        single_prompt = create_single_prompt(original, language)
        
        # Store sample prompts (first 3)
        if len(sample_prompts) < 3:
            sample_prompts.append({
                "type": f"paragraph_{i + 1}",
                "prompt": single_prompt,
                "original": original
            })
        
        # Add to jobs list
        jobs.append((i, para, single_prompt))
        
        i += 1

    total_paragraphs = len(jobs)
    logs.append(f"[QUEUE] Prepared {total_paragraphs} paragraphs for translation")
    logs.append(f"[PROCESSING] Starting parallel API requests (paragraph-by-paragraph)...")
    
    # Create thread pool executor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    
    # Create tasks with proper indexing for real-time progress tracking
    task_futures = []
    for idx, (para_idx, para, prompt) in enumerate(jobs):
        task = asyncio.create_task(call_gemini_async(executor, client, prompt, model, logs))
        task_futures.append((idx, task))
    
    # Process tasks with real-time progress tracking
    completed_count = 0
    ordered_results = [None] * len(task_futures)
    
    # Use asyncio.as_completed for real-time progress
    for completed_task in asyncio.as_completed([task for _, task in task_futures]):
        result = await completed_task
        
        # Find which task completed by matching the task object
        task_index = None
        for i, (idx, task) in enumerate(task_futures):
            if task == completed_task:
                task_index = idx
                break
        
        if task_index is not None:
            ordered_results[task_index] = result
        
        completed_count += 1
        
        # Update progress in real-time
        if progress_callback:
            progress_callback(completed_count / len(task_futures))
        
        # Update token tracking in real-time
        if result and token_callback:
            tracker.add_request_tokens(
                result.get('input_tokens', 0),
                result.get('output_tokens', 0)
            )
            token_callback(tracker.get_stats())
    
    results = ordered_results
    
    logs.append(f"[COMPLETE] All API requests finished")
    logs.append(f"[PROCESSING] Applying translations to document...")
    
    # Apply translations to document in original order
    for job_data, result in zip(jobs, results):
        para_idx, para, prompt = job_data
        original = para.text.strip()
        
        if result and 'text' in result:
            translated = result['text']
            
            # Clean up triple quotes if present
            if translated.startswith('"""') and translated.endswith('"""'):
                translated = translated[3:-3].strip()
            elif translated.startswith('"""'):
                translated = translated[3:].strip()
            
            translated = sanitize_response(translated)
            
            if translated:
                translated_content.append(translated)
                
                # Update paragraph text in memory
                for run in para.runs:
                    run.text = ""
                if para.runs:
                    para.runs[0].text = translated
                else:
                    para.add_run(translated)
            else:
                # Fallback for empty translation
                fallback_text = f"[Translation failed for paragraph {para_idx}]"
                translated_content.append(fallback_text)
                logs.append(f"[WARNING] Empty translation for paragraph {para_idx}")
        else:
            # API call failed
            fallback_text = f"[API call failed for paragraph {para_idx}]"
            translated_content.append(fallback_text)
            logs.append(f"[ERROR] API call failed for paragraph {para_idx}")
    
    # Save document to memory buffer instead of disk
    output_buffer = io.BytesIO()
    doc.save(output_buffer)
    output_buffer.seek(0)  # Reset buffer position to start
    
    final_stats = tracker.get_stats()
    
    logs.append(f"[SAVE] Document saved to memory buffer")
    logs.append(f"[TOKENS] Final usage - Input: {final_stats['total_input_tokens']}, Output: {final_stats['total_output_tokens']}, Total: {final_stats['total_tokens']}")
    logs.append(f"[DONE] Translation complete!")
    
    executor.shutdown(wait=False)
    
    # Return document bytes instead of file path
    return output_buffer.getvalue(), translated_content, logs, sample_prompts, final_stats

# --- Streamlit UI ---
st.title("📘 Easy Translate: Paragraph-by-Paragraph Translation")
st.caption("🎯 Individual Paragraph Processing • 📊 Real-time Token Tracking • 💾 In-Memory Processing")

uploaded_file = st.file_uploader("Upload a .docx or .pdf manuscript", type=["pdf", "docx"])

# Model and language selection
col1, col2 = st.columns([2, 2])
with col1:
    target_language = st.selectbox("Select output language", ["Contemporary English", "Spanish", "German", "Dutch"])
with col2:
    gemini_model = st.selectbox(
        "Select Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        help="Choose between Flash (faster, cheaper) or Thinking (enhanced reasoning)"
    )

# Info about processing mode
st.info(f"📄 **Paragraph-by-Paragraph Mode**: Each paragraph is sent as a separate API request. Up to {MAX_CONCURRENT_REQUESTS} requests run in parallel.")
st.info("💾 **In-Memory Processing**: No files are saved to local storage - perfect for secure environments.")

if uploaded_file:
    base_name = os.path.splitext(uploaded_file.name)[0]
    lang_suffix = target_language.lower().replace(" ", "_")
    final_filename = f"{base_name}_{lang_suffix}_nobatch.docx"

    if st.button("🚀 Start Translation"):
        if not GEMINI_API_KEY:
            st.error("❌ Cannot translate: GEMINI_API_KEY is not configured!")
        else:
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
                        st.metric("📋 API Requests", f"{stats['total_requests']}")
                    with col2:
                        st.metric("⏱️ Elapsed Time", f"{stats['elapsed_time']:.1f}s")
                    with col3:
                        avg_tokens = stats['avg_tokens_per_request']
                        st.metric("📊 Avg Tokens/Request", f"{avg_tokens:.0f}")

            with st.spinner("🔄 Translating paragraph-by-paragraph (in memory)..."):
                try:
                    document_bytes, translated_content, logs, sample_prompts, final_stats = asyncio.run(
                        translate_docx_paragraph_by_paragraph(
                            uploaded_file, 
                            target_language, 
                            gemini_model, 
                            update_progress, 
                            update_token_dashboard
                        )
                    )

                    progress_bar.empty()
                    st.success("✅ Translation completed! (Processed paragraph-by-paragraph in memory)")
                    
                    # Final analytics summary
                    st.subheader("🎯 Final Analytics Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💰 Total Cost Est.", f"${final_stats['total_tokens'] * 0.000001:.4f}")
                    with col2:
                        efficiency = final_stats['total_output_tokens'] / final_stats['total_input_tokens'] if final_stats['total_input_tokens'] > 0 else 0
                        st.metric("⚡ Efficiency Ratio", f"{efficiency:.2f}x")
                    with col3:
                        requests_per_min = (final_stats['total_requests'] / final_stats['elapsed_time'] * 60) if final_stats['elapsed_time'] > 0 else 0
                        st.metric("🚀 Requests/Min", f"{requests_per_min:.1f}")
                    with col4:
                        st.metric("📄 Total Paragraphs", f"{final_stats['total_requests']}")
                    
                    # Performance insights
                    avg_request_time = final_stats['elapsed_time'] / final_stats['total_requests'] if final_stats['total_requests'] > 0 else 0
                    st.info(
                        f"🚀 **Performance Summary:** "
                        f"Processed {final_stats['total_requests']} paragraphs in {final_stats['elapsed_time']:.1f}s "
                        f"(avg {avg_request_time:.2f}s per request) • "
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
                        with st.expander("📝 Sample Prompts", expanded=False):
                            for i, sample in enumerate(sample_prompts, 1):
                                st.markdown(f"**Paragraph #{i} - {sample['type'].replace('_', ' ').title()}**")
                                st.text_area(
                                    f"Original Text #{i}:",
                                    value=sample['original'],
                                    height=80,
                                    key=f"original_{i}"
                                )
                                st.text_area(
                                    f"Full Prompt #{i}:",
                                    value=sample['prompt'],
                                    height=300,
                                    key=f"prompt_{i}"
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
                        help="Download the translated document as a Word file (processed entirely in memory)"
                    )
                
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Translation failed: {str(e)}")
                    st.exception(e)
