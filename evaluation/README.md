# Evaluation scripts

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 1. Multiple Choice - Single Answer (MCQU)
- **Script**: `evaluation/mcqu/eval_mcqu.py`
- **Metric**: Accuracy

```bash
python ./mcqu/eval_mcqu.py
```

## 2. Multiple Choice - Multiple Answers (MCQM)
- **Script**: `evaluation/mcqm/eval_mcqm.py`
- **Metrics**: 
  - Exact Match Ratio (EMR)
  - Hamming Score

```bash
python ./mcqm/eval_mcqm.py
```

## 3. Open-Ended QA (OEQ)

### Lexical and contextual embedding-based metrics
- **Script**: `evaluation/oeq/eval_oeq.py`
- **Metrics**:
  - ROUGE
  - BLEU
  - BERTScore

```bash
python ./oeq/eval_oeq.py
```


### 4. LLM-as-Judge metric
- **Scripts**:
  - `create_prompt_llm_judge.py`: Generates prompts for LLMs via API (e.g., GPT).
  - `eval_llm_judge.py`: Computes final evaluation scores based on LLM outputs.

 




