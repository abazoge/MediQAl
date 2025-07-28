# MediQAl: A French Medical Question Answering Dataset for Knowledge and Reasoning Evaluation

MediQAl is a French medical question answering dataset designed to evaluate the capabilities of language models in factual medical recall and clinical reasoning. It includes **32,603 questions** sourced from French medical examinations across **41 medical subjects**.

The dataset contains three tasks:
- MCQU: Multiple-Choice Questions with a Unique correct answer
- MCQM: Multiple-Choice Questions with Multiple correct answers
- OEQ: Open-Ended Questions with Short Answers

Each question is labeled as either "Understanding" or "Reasoning", enabling analysis of the cognitive capabilities of language models.

The dataset is available under CC-BY-4.0 licence on [HuggingFace](https://huggingface.co/datasets/ANR-MALADES/MediQAl).

## 📁 Repository Structure

```shell
MediQAl/
├── evaluation        # Evaluation scripts for each subset of MediQAl
    ├── mcqu 			
    ├── mcqm
    └── oeq
├── inference        # Scripts for running inference on evaluated models
└── sft        # Script for finetuning models on MediQAl
```

## 📖 Citation

```bibtex
@misc{bazoge2025mediqal,
      title={MediQAl: A French Medical Question Answering Dataset for Knowledge and Reasoning Evaluation}, 
      author={Adrien Bazoge},
      year={2025},
      eprint={-},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={-}, 
}
```
