# Supervised Fine-tuning on MediQAl

## 🧾 Formatting Input

Before fine-tuning, format the input questions using the provided script. This will generate the training and validation datasets.

```bash
python create_dataset_SFT.py
```

## 🛠 Fine-Tuning

Use the `sft_trainer.py` script to fine-tune any model on the MediQAl dataset. Both full fine-tuning and LoRA are supported.

Example for full fine-tuning of BioMistral-7B:
```bash
python sft_trainer.py \
    --model_name_or_path ../models/BioMistral_BioMistral-7B \
    --learning_rate 2.0e-5 \
    --num_train_epochs 5 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '</s>' \
    --eval_strategy steps \
    --eval_steps 500 \
    --output_dir BioMistral_BioMistral-7B-SFT \
```
