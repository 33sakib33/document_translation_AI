import os
import re
import asyncio
import time
import streamlit as st
from docx import Document
from tempfile import NamedTemporaryFile
from ollama import chat, ChatResponse
from concurrent.futures import ThreadPoolExecutor
from pdf_utils import convert_pdf_to_docx_adobe

# --- CONFIGURATION ---
# Load PDF Services credentials
try:
    PDF_SERVICES_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID") or st.secrets.get("PDF_SERVICES_CLIENT_ID")
    PDF_SERVICES_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET") or st.secrets.get("PDF_SERVICES_CLIENT_SECRET")
except Exception:
    PDF_SERVICES_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID")
    PDF_SERVICES_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET")

MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_CONCURRENT_REQUESTS = 3  # Lower for local inference to avoid overload
DEFAULT_MODEL = "qwen3:8b"  # Change to your preferred Ollama model

# --- Prompt Template ---
PROMPT_TEMPLATE = """
Here is a passage of a manuscript that we want to translate into {language} to make it easily readable for everyone. This is the original English version of one of the passages:
Can you rephrase this text in fluent and natural {language}? Please rewrite it and do not plagiarize. Do not give an answer, only give the translated text. 
Make sure that the reading experience "flows", for example by not using the same words & sentence structures too often. Only translate, do not mention anything else. Please format the text properly without separating lines between the passages and exactly in the same structure it was so I can easily copy/paste it entirely into the manuscript of the book. You can remove numbers if that makes the reading experience better. You must translate/rewrite everything exactly and do not shorten it. Keep the quotes in their formatted way if applicable.
This is important for my career: Please try to translate everything sentence-by-sentence and do not make it shorter!
Important: Please Translate everything sentence-by-sentence in fluent {language} and do not make it shorter!
So do not skip translating any sentence line-by-line into {language} and do not make the translation shorter. This is crucial!
This is crucial: When you encounter general terminology such as "chapter," "the end,", "introduction" etc., or phrases that are widely used across books, do not vary or rephrase them—this is to ensure consistency. Always translate the text into the targetted language
Always use the most widely accepted or officially published book title, chapter name, or term if multiple synonyms are possible — e.g., use Meditations, not Reflections; Letters from a Stoic, not Stoic Letters. "Introduction", not "getting started"
Also, always choose the most commonly published or recognized title, name or term in the target language — e.g., De Gedaanteverwisseling, not Metamorfose in Dutch. "Arme Peter" ipv "poor Peter" or "Armzalige Peter"
Important: Only give the exact translation; do not answer anything else, do not provide any commentary nor any introductory text(i.e. here are...), such as "Here's a concise, modern rewrite," etc./no_think.

Original Passage:
\"\"\"
{chunk}
\"\"\"
"""

# --- Evaluation Template ---
EVALUATION_TEMPLATE = """You are a translation evaluator. 
Your task is to judge the quality of a machine translation with respect to:
1. Meaning preservation (adequacy) — Does the translation fully convey the meaning of the source text?
2. Fluency — Is the translation grammatically correct and natural in the target language?
3. Prompt compliance — Does the translation follow the specific instructions from the prompt (tone, style, terminology, formatting, etc.)?
4. Entity and number fidelity — Are named entities, dates, numbers, and key terms preserved or correctly translated?

Instructions:
- Carefully compare the **Source Text** and **Translation** in light of the **Prompt**.
- First analyze adequacy, fluency, compliance, and fidelity.
- Then provide a structured JSON response with numeric scores and flags.

Scoring guidelines:
- adequacy: integer 0–100 (0 = completely wrong, 100 = perfect meaning preservation)
- fluency: integer 1–5 (1 = broken/ungrammatical, 5 = native-quality)
- prompt_compliance: 0 or 1 (0 = fails, 1 = passes)
- entity_fidelity: float 0–1 (fraction of correctly preserved entities/numbers)
- overall: "accept" if adequacy ≥ 70, fluency ≥ 3, prompt_compliance = 1, entity_fidelity ≥ 0.9; otherwise "reject"

Output ONLY valid JSON in this format:

{{
  "adequacy": <int>,
  "fluency": <int>,
  "prompt_compliance": <0 or 1>,
  "entity_fidelity": <float>,
  "errors": [ "list of short error categories, e.g. 'mistranslation', 'number mismatch', 'style violation'" ],
  "overall": "<accept or reject>"
}}

---

Prompt: {prompt}

Source Text: {source_text}

Translation: {translation}
"""

# --- Helper Functions ---
def is_meaningful_text(text):
    cleaned = re.sub(r'[\W_]+', '', text)
    return bool(cleaned.strip())

def is_decorative_only(text):
    stripped = text.strip()
    return not stripped or re.fullmatch(r"[^\w\s]+", stripped) or re.fullmatch(r"[A-Z]", stripped)


def sanitize_response(text: str) -> str:
    """Remove any <think>...</think> tokens and trim the response.

    This removes multi-line content inside <think> tags (case-insensitive).
    """
    if not text:
        return text
    text = re.sub(r'(?is)<think>.*?</think>', '', text)
    text = re.sub(r'(?i)</?think>', '', text)
    return text.strip()

def evaluate_translation(model, prompt, source_text, translation, log_list=None):
    """Evaluate translation quality using model-as-judge approach.
    
    Returns:
        dict: Evaluation results with adequacy, fluency, prompt_compliance, entity_fidelity, errors, overall
    """
    import json
    
    eval_prompt = EVALUATION_TEMPLATE.format(
        prompt=prompt,
        source_text=source_text,
        translation=translation
    )
    
    if log_list is not None:
        log_list.append("[EVAL] Starting translation quality evaluation...")
    
    try:
        response: ChatResponse = chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': eval_prompt,
            }],
            options={
                'temperature': 0.1,  # Low temperature for consistent evaluation
            }
        )
        
        result = response.message.content.strip()
        
        if log_list is not None:
            log_list.append(f"[EVAL] Raw response: {result[:200]}...")
        
        # Extract JSON from response (might have markdown code blocks)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            eval_result = json.loads(json_str)
            
            if log_list is not None:
                log_list.append(f"[EVAL] Parsed result: {eval_result}")
            
            return eval_result
        else:
            if log_list is not None:
                log_list.append("[EVAL] Failed to parse JSON from response")
            return None
            
    except Exception as e:
        if log_list is not None:
            log_list.append(f"[EVAL] Evaluation failed: {str(e)}")
        return None

def call_ollama_api(model, prompt, log_list=None):
    """
    Synchronous function to call Ollama with retries
    """
    for attempt in range(MAX_RETRIES):
        try:
            if log_list is not None:
                log_entry = f"[API Call] Attempt {attempt + 1}/{MAX_RETRIES} - Model: {model}"
                log_list.append(log_entry)
            
            response: ChatResponse = chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ]
            )
            
            # Access the response content
            result = response.message.content.strip()
            
            if log_list is not None:
                log_list.append(f"[Success] Received response ({len(result)} chars)")
            
            return result
            
        except Exception as e:
            if log_list is not None:
                log_list.append(f"[Error] Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                import time
                time.sleep(RETRY_DELAY)
    
    if log_list is not None:
        log_list.append("[Failed] All retry attempts exhausted")
    return None

async def call_ollama_async(executor, model, prompt, log_list=None):
    """
    Async wrapper for Ollama API call
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, call_ollama_api, model, prompt, log_list)
    return result

async def translate_docx_async(docx_path, output_path, language, model, progress_callback=None, enable_evaluation=False):
    """
    Translate a DOCX file using Ollama with optional quality evaluation
    """
    start_time = time.time()
    
    doc = Document(docx_path)
    paragraphs = doc.paragraphs

    jobs = []
    translated_content = []
    logs = []
    sample_prompts = []
    evaluation_results = []
    
    # Analytics tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_input_words = 0
    total_output_words = 0
    
    logs.append(f"[START] Translation started for language: {language}")
    logs.append(f"[INFO] Document has {len(paragraphs)} total paragraphs")
    logs.append(f"[INFO] Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    logs.append(f"[INFO] Using Ollama model: {model}")
    logs.append(f"[INFO] Running locally - no API key needed! 🎉")
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        original = para.text.strip()

        # Skip single uppercase letters followed by uppercase
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

        # Skip short lines unless clearly uppercase or marked as a heading
        if word_count <= 1:
            if not original.isupper() and not is_heading:
                i += 1
                continue
            elif is_heading and language.lower() == "contemporary english":
                prompt = f"Rewrite this heading in contemporary English, keeping it short and clear:\n\"\"\"\n{original}\"\"\""
                jobs.append((i, para, prompt))
                logs.append(f"[HEADING] Queued heading translation: '{original[:50]}...'")
                if len(sample_prompts) < 3:
                    sample_prompts.append({"type": "heading", "prompt": prompt, "original": original})
                i += 1
                continue

        prompt = PROMPT_TEMPLATE.format(chunk=original, language=language)
        jobs.append((i, para, prompt))
        
        # Track input statistics
        total_input_words += len(original.split())
        total_input_tokens += len(prompt.split())  # Rough approximation
        
        if len(sample_prompts) < 3:
            sample_prompts.append({"type": "paragraph", "prompt": prompt, "original": original})
        
        i += 1

    total = len(jobs)
    logs.append(f"[QUEUE] Prepared {total} paragraphs for translation")
    logs.append(f"[PROCESSING] Starting parallel requests to local Ollama...")
    
    # Create thread pool executor for parallel requests
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    
    # Process translations with progress tracking (preserve original order)
    completed = 0
    
    # Create tasks and remember their original indices
    tasks = [
        call_ollama_async(executor, model, prompt, logs)
        for _, _, prompt in jobs
    ]
    
    # Use asyncio.gather to maintain order and get progress
    # This is simpler and preserves the original order automatically
    results = await asyncio.gather(*tasks)
    
    # Update progress bar to 100% after all complete
    if progress_callback:
        progress_callback(1.0)
    
    logs.append(f"[COMPLETE] All requests finished")
    logs.append(f"[PROCESSING] Applying translations to document...")
    
    # Apply translations to document in original order
    for (i, para, prompt), translated in zip(jobs, results):
        if translated:
            # Clean up triple quotes if present
            if translated.startswith('"""') and translated.endswith('"""'):
                translated = translated[3:-3].strip()
            elif translated.startswith('"""'):
                translated = translated[3:].strip()
            elif translated.endswith('"""'):
                translated = translated[:-3].strip()

            # Sanitize response
            translated = sanitize_response(translated)

            translated_content.append(translated)
            
            # Track output statistics
            total_output_words += len(translated.split())
            total_output_tokens += len(translated.split())  # Rough approximation
            
            # Evaluate translation quality if enabled
            if enable_evaluation and len(evaluation_results) < 10:  # Limit to first 10 for performance
                source_text = para.text.strip()
                eval_result = evaluate_translation(model, prompt, source_text, translated, logs)
                if eval_result:
                    evaluation_results.append({
                        'paragraph_index': i,
                        'source': source_text[:100] + ('...' if len(source_text) > 100 else ''),
                        'translation': translated[:100] + ('...' if len(translated) > 100 else ''),
                        'evaluation': eval_result
                    })

            # Update paragraph text
            for run in para.runs:
                run.text = ""
            if para.runs:
                para.runs[0].text = translated
            else:
                para.add_run(translated)

    doc.save(output_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logs.append(f"[SAVE] Document saved to: {output_path}")
    logs.append(f"[DONE] Translation complete!")
    logs.append(f"[TIMING] Total execution time: {elapsed_time:.2f} seconds")
    print(f"\n✅ Translated file saved to: {output_path}")
    
    executor.shutdown(wait=False)
    
    # Create analytics dictionary
    analytics = {
        'execution_time': elapsed_time,
        'total_paragraphs': len(jobs),
        'total_input_words': total_input_words,
        'total_output_words': total_output_words,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'words_per_second': total_input_words / elapsed_time if elapsed_time > 0 else 0,
        'paragraphs_per_minute': (len(jobs) / elapsed_time * 60) if elapsed_time > 0 else 0
    }
    
    return output_path, translated_content, logs, sample_prompts, evaluation_results, analytics


# --- Streamlit UI ---
st.title("📘 Easy Translate: Manuscript Translator (Ollama)")
st.caption("🏠 Powered by Local Ollama - No API Keys Needed!")

# Model selection
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Upload a .docx or .pdf manuscript", type=["pdf", "docx"])
with col2:
    st.markdown("####")  # Spacing
    
# Language selection
target_language = st.selectbox("Select output language", ["Contemporary English", "Spanish", "German", "Dutch"])

# Use default model from configuration
ollama_model = DEFAULT_MODEL

# Evaluation option
enable_evaluation = st.checkbox(
    "🔍 Evaluate Translation Quality (Model-as-Judge)",
    value=False,
    help="Use LLM to evaluate translation quality on adequacy, fluency, prompt compliance, and entity fidelity. Evaluates first 10 paragraphs."
)

# Info box about Ollama
with st.expander("ℹ️ About Ollama Setup", expanded=False):
    st.markdown(f"""
    **Ollama runs locally on your machine - no internet or API keys needed!**
    
    **Current Model:** `{DEFAULT_MODEL}`
    
    **Setup Instructions:**
    1. Install Ollama: [https://ollama.ai](https://ollama.ai)
    2. Pull the model: `ollama pull {DEFAULT_MODEL}`
    3. Start Ollama service (usually auto-starts)
    4. Use this app!
    
    **To change the model:**
    Edit `DEFAULT_MODEL` at the top of `app_ollama.py`
    
 
    **Check installed models:** `ollama list`
    """)

if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[-1].lower()

    with NamedTemporaryFile(delete=False, suffix=ext) as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input.flush()
        input_path = temp_input.name

    base_name = os.path.splitext(uploaded_file.name)[0]
    lang_suffix = target_language.lower().replace(" ", "_")
    final_filename = f"{base_name}_{lang_suffix}.docx"
    output_path = os.path.join(".", final_filename)

    if ext == ".pdf":
        if not PDF_SERVICES_CLIENT_ID or not PDF_SERVICES_CLIENT_SECRET:
            st.warning("⚠️ PDF Services credentials not configured. PDF conversion may fail.")
        
        with st.spinner("🔄 Converting PDF to DOCX using Adobe..."):
            try:
                converted_docx = convert_pdf_to_docx_adobe(input_path)
                docx_path = converted_docx
            except Exception as e:
                st.error(f"❌ PDF conversion failed: {str(e)}")
                st.stop()
    else:
        docx_path = input_path

    if st.button("🚀 Translate Now"):
        progress_bar = st.progress(0)

        def update_progress(pct):
            progress_bar.progress(pct)

        with st.spinner(f"Translating with Ollama"):
            try:
                output_path, translated_content, logs, sample_prompts, evaluation_results, analytics = asyncio.run(
                    translate_docx_async(docx_path, output_path, target_language, ollama_model, update_progress, enable_evaluation)
                )

                progress_bar.empty()
                st.success("✅ Translation completed!")
                
                # Analytics Dashboard
                st.subheader("📊 Translation Analytics")
                
                # Main metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "⏱️ Execution Time",
                        f"{analytics['execution_time']:.1f}s",
                        help="Total time to translate the document"
                    )
                with col2:
                    st.metric(
                        "📄 Paragraphs",
                        f"{analytics['total_paragraphs']}",
                        delta=f"{analytics['paragraphs_per_minute']:.1f}/min",
                        help="Total paragraphs translated"
                    )
                with col3:
                    st.metric(
                        "📝 Input Words",
                        f"{analytics['total_input_words']:,}",
                        help="Total words in source document"
                    )
                with col4:
                    st.metric(
                        "📗 Output Words",
                        f"{analytics['total_output_words']:,}",
                        delta=f"{analytics['words_per_second']:.1f}/s",
                        help="Total words in translated document"
                    )
                
                # Token usage metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "🎯 Input Tokens",
                        f"{analytics['total_input_tokens']:,}",
                        help="Approximate tokens sent to model (including prompts)"
                    )
                with col2:
                    st.metric(
                        "🎯 Output Tokens",
                        f"{analytics['total_output_tokens']:,}",
                        help="Approximate tokens received from model"
                    )
                with col3:
                    total_tokens = analytics['total_input_tokens'] + analytics['total_output_tokens']
                    st.metric(
                        "🎯 Total Tokens",
                        f"{total_tokens:,}",
                        help="Total tokens processed"
                    )
                
                # Performance summary in info box
                avg_time_per_para = analytics['execution_time'] / analytics['total_paragraphs'] if analytics['total_paragraphs'] > 0 else 0
                compression_ratio = analytics['total_output_words'] / analytics['total_input_words'] if analytics['total_input_words'] > 0 else 1
                
                st.info(
                    f"📈 **Performance Summary:** "
                    f"Averaged {avg_time_per_para:.2f}s per paragraph • "
                    f"Translation ratio: {compression_ratio:.2f}x • "
                    f"Throughput: {analytics['words_per_second']:.1f} words/sec"
                )
                
                st.divider()
                
                # Display logs section
                st.subheader("📋 Translation Logs")
                with st.expander("🔍 View Processing Logs", expanded=False):
                    log_text = "\n".join(logs)
                    st.code(log_text, language="log")
                
                # Display sample prompts
                if sample_prompts:
                    with st.expander("📝 Sample Prompts (First 3)", expanded=True):
                        for i, sample in enumerate(sample_prompts, 1):
                            st.markdown(f"**Sample #{i} - {sample['type'].title()}**")
                            st.text_area(
                                f"Original Text #{i}:",
                                value=sample['original'],
                                height=100,
                                key=f"original_{i}"
                            )
                            st.text_area(
                                f"Full Prompt #{i} sent to Ollama:",
                                value=sample['prompt'],
                                height=300,
                                key=f"prompt_{i}"
                            )
                            if i < len(sample_prompts):
                                st.divider()
                
                # Display evaluation results
                if enable_evaluation and evaluation_results:
                    st.divider()
                    st.subheader("🎯 Translation Quality Evaluation")
                    
                    # Calculate summary statistics
                    accept_count = sum(1 for r in evaluation_results if r['evaluation']['overall'] == 'accept')
                    reject_count = sum(1 for r in evaluation_results if r['evaluation']['overall'] == 'reject')
                    avg_adequacy = sum(r['evaluation']['adequacy'] for r in evaluation_results) / len(evaluation_results)
                    avg_fluency = sum(r['evaluation']['fluency'] for r in evaluation_results) / len(evaluation_results)
                    avg_entity = sum(r['evaluation']['entity_fidelity'] for r in evaluation_results) / len(evaluation_results)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("✅ Accepted", f"{accept_count}/{len(evaluation_results)}")
                    with col2:
                        st.metric("❌ Rejected", f"{reject_count}/{len(evaluation_results)}")
                    with col3:
                        st.metric("📊 Avg Adequacy", f"{avg_adequacy:.1f}/100")
                    with col4:
                        st.metric("💬 Avg Fluency", f"{avg_fluency:.1f}/5")
                    
                    # Detailed results
                    with st.expander("📋 Detailed Evaluation Results", expanded=False):
                        for idx, result in enumerate(evaluation_results, 1):
                            eval_data = result['evaluation']
                            status_emoji = "✅" if eval_data['overall'] == 'accept' else "❌"
                            
                            st.markdown(f"### {status_emoji} Paragraph {result['paragraph_index']} - {eval_data['overall'].upper()}")
                            
                            # Create metric columns
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                adequacy_color = "🟢" if eval_data['adequacy'] >= 70 else "🔴"
                                st.markdown(f"{adequacy_color} **Adequacy:** {eval_data['adequacy']}/100")
                            with col2:
                                fluency_color = "🟢" if eval_data['fluency'] >= 3 else "🔴"
                                st.markdown(f"{fluency_color} **Fluency:** {eval_data['fluency']}/5")
                            with col3:
                                compliance_color = "🟢" if eval_data['prompt_compliance'] == 1 else "🔴"
                                st.markdown(f"{compliance_color} **Compliance:** {'✓' if eval_data['prompt_compliance'] == 1 else '✗'}")
                            with col4:
                                entity_color = "🟢" if eval_data['entity_fidelity'] >= 0.9 else "🔴"
                                st.markdown(f"{entity_color} **Entity Fidelity:** {eval_data['entity_fidelity']:.2f}")
                            
                            # Show errors if any
                            if eval_data.get('errors'):
                                st.markdown(f"**⚠️ Issues:** {', '.join(eval_data['errors'])}")
                            
                            # Show text snippets
                            with st.expander("View Source & Translation"):
                                st.text_area("Source:", result['source'], height=80, key=f"eval_src_{idx}")
                                st.text_area("Translation:", result['translation'], height=80, key=f"eval_trans_{idx}")
                            
                            if idx < len(evaluation_results):
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
                
                # Download section
                st.subheader("💾 Download Options")
                with open(output_path, "rb") as f:
                    st.download_button(
                        "📥 Download Translated DOCX", 
                        f, 
                        file_name=final_filename, 
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        help="Download the translated document as a Word file"
                    )
            
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Translation failed: {str(e)}")
                st.markdown("""
                **Troubleshooting:**
                - Make sure Ollama is running: `ollama serve`
                - Check if the model is installed: `ollama list`
                - Pull the model if needed: `ollama pull {}`
                """.format(ollama_model))
                st.exception(e)
