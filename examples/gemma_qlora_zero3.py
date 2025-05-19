import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from fastDP import PrivacyEngine
import deepspeed
import os


def get_args():
    parser = argparse.ArgumentParser(description="DP QLoRA fine-tuning with Gemma")
    parser.add_argument("--model_name", default="google/gemma-8b", help="Gemma model name")
    parser.add_argument("--dataset", required=True, help="Path or name of the training dataset")
    parser.add_argument("--output_dir", default="./dp_qlora_out")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--target_epsilon", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quant_config,
    )

    if os.path.exists(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset)["train"]
    else:
        dataset = load_dataset(args.dataset, split="train")

    def tokenize(examples):
        out = tokenizer(examples["text"])
        out["labels"] = out["input_ids"].copy()
        return out

    train_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        deepspeed={"zero_optimization": {"stage": 3}},
        fp16=True,
    )

    # Attach DP engine
    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.per_device_train_batch_size,
        sample_size=len(train_dataset),
        epochs=args.num_train_epochs,
        target_epsilon=args.target_epsilon,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    num_steps = (len(train_dataset) // training_args.per_device_train_batch_size)
    num_steps = max(num_steps, 1) * training_args.num_train_epochs
    trainer.create_optimizer_and_scheduler(num_training_steps=num_steps)
    privacy_engine.attach(trainer.optimizer)

    trainer.train()


if __name__ == "__main__":
    main()
