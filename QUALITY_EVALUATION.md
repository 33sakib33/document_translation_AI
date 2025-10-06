# Translation Quality Evaluation Feature

## Overview

The translation apps now include an optional quality evaluation system that analyzes translations using three key metrics:

1. **Semantic Similarity** - How well the translation preserves meaning
2. **Entity Preservation** - Whether numbers and entities are correctly maintained
3. **Fluency** - How natural and grammatically correct the translation sounds

## How It Works

### Models Used

- **LaBSE (Language-agnostic BERT Sentence Embeddings)**: Cross-lingual sentence embeddings for semantic similarity
- **GPT-2**: Language model for fluency evaluation via perplexity scoring

### Scoring System

Each translation paragraph receives:

- **Semantic Similarity Score** (0-1): Cosine similarity between source and translation embeddings
- **Entity Preservation Score** (0-1): Ratio of numbers/entities preserved
- **Fluency Score** (0-1): Inverse perplexity normalized score
- **Overall Score**: Average of the three metrics

### Decision Thresholds

- **✅ Accept**: Score ≥ 0.75 (High quality)
- **⚠️ Review**: Score 0.50-0.74 (May need review)
- **❌ Reject**: Score < 0.50 (Low quality, needs retranslation)

## Usage

### Enable Quality Check

In the UI, check the **"Enable Quality Check"** checkbox before translating.

⚠️ **Note**: Quality evaluation adds processing time. For large documents, it may significantly slow down the translation.

### Viewing Results

After translation completes, you'll see:

1. **Quality Report Dashboard**:

   - Average quality score across all paragraphs
   - Count of accepted/review/rejected paragraphs
   - Visual breakdown

2. **Detailed Breakdown** (expandable):
   - Individual scores for first 10 paragraphs
   - Breakdown of each metric per paragraph

## Installation

The quality evaluation feature requires additional dependencies:

```bash
pip install sentence-transformers transformers torch numpy
```

These are already included in `requirements.txt`.

## Performance Considerations

- **First run**: Models will be downloaded (can take a few minutes)
- **Memory**: Requires ~2GB RAM for models
- **Speed**: Adds ~2-5 seconds per paragraph
- **Recommendation**: Use for final quality checks or important documents

## Implementation Details

### Semantic Similarity

- Uses LaBSE embeddings (multilingual)
- Computes cosine similarity between source and target
- Works across different languages

### Entity Preservation

- Extracts numbers using regex
- Checks if source numbers appear in translation
- Calculates percentage preserved

### Fluency

- Tokenizes translation with GPT-2
- Calculates perplexity (lower = more fluent)
- Normalizes to 0-1 score

## Limitations

1. **GPT-2 is English-centric**: Fluency scores may be less accurate for non-English translations
2. **Entity detection**: Only detects numbers, not named entities
3. **Context-independent**: Evaluates each paragraph separately
4. **Model limitations**: Quality is bounded by evaluation model capabilities

## Future Improvements

Potential enhancements:

- Use target-language-specific fluency models
- Add BLEU/METEOR reference-free metrics
- Include named entity recognition
- Add readability scores
- Parallel evaluation to reduce overhead

## Files Modified

- `app_gemini.py`: Added quality evaluation
- `requirements.txt`: Added sentence-transformers, transformers, torch, numpy
- `QUALITY_EVALUATION.md`: This documentation

## Example Output

```
Translation Quality Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average Score: 78.3%
✅ Accepted: 45/50
⚠️ Review: 4/50
❌ Rejected: 1/50

Detailed Quality Breakdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Paragraph 1 - Score: 82.1%
- Semantic Similarity: 87.3%
- Entity Preservation: 100.0%
- Fluency: 59.0%

⚠️ Paragraph 2 - Score: 68.5%
- Semantic Similarity: 71.2%
- Entity Preservation: 75.0%
- Fluency: 59.3%
```

## Troubleshooting

**Models not downloading?**

- Check internet connection
- Ensure sufficient disk space (~2GB)
- Try manually: `from sentence_transformers import SentenceTransformer; SentenceTransformer('LaBSE')`

**Out of memory?**

- Disable quality check for large documents
- Process in smaller batches
- Upgrade RAM or use CPU-only mode

**Slow evaluation?**

- First run downloads models (one-time)
- GPU acceleration recommended for large docs
- Consider disabling for drafts, enabling for final versions
