"""Train a small causal LM to predict legal chess moves from board states."""

from __future__ import annotations

import argparse
from dataclasses import replace

from gpt_chess.config import (
    DEFAULT_CONFIG,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainerConfig,
)
from gpt_chess.data import tokenize_dataset
from gpt_chess.modeling import attach_lora_adapter, load_base_model_and_tokenizer


def train(config: ExperimentConfig = DEFAULT_CONFIG):
    """Run the configured fine-tuning job and save the model/tokenizer."""

    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tokenizer, mapper = load_base_model_and_tokenizer(config.model)
    model = attach_lora_adapter(model, config.model)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    raw_dataset = load_dataset(
        config.data.dataset_name,
        split=config.data.dataset_split,
    )
    train_dataset = tokenize_dataset(raw_dataset, mapper=mapper, config=config.data)
    print(f"Training examples extracted: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=config.model.output_dir,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        learning_rate=config.trainer.learning_rate,
        num_train_epochs=config.trainer.num_train_epochs,
        logging_steps=config.trainer.logging_steps,
        optim=config.trainer.optim,
        report_to=config.trainer.report_to,
        save_strategy=config.trainer.save_strategy,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.model.save_pretrained(config.model.output_dir)
    tokenizer.save_pretrained(config.model.output_dir)
    return train_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_CONFIG.model.model_id)
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.model.output_dir)
    parser.add_argument("--dataset-name", default=DEFAULT_CONFIG.data.dataset_name)
    parser.add_argument("--dataset-split", default=DEFAULT_CONFIG.data.dataset_split)
    parser.add_argument(
        "--position-policy",
        choices=["all_plies", "final_ply"],
        default=DEFAULT_CONFIG.data.position_policy,
        help="Use all pre-move boards, or only the final pre-move board per game.",
    )
    parser.add_argument(
        "--no-fen-metadata",
        action="store_true",
        help="Use only the 71-token board string inside chess tags.",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Fine-tune the full model instead of attaching a LoRA adapter.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG.trainer.num_train_epochs,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG.trainer.per_device_train_batch_size,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG.trainer.learning_rate,
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    model = replace(
        ModelConfig(),
        model_id=args.model_id,
        output_dir=args.output_dir,
        use_lora=not args.no_lora,
    )
    data = replace(
        DataConfig(),
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        include_fen_metadata=not args.no_fen_metadata,
        position_policy=args.position_policy,
    )
    trainer = replace(
        TrainerConfig(),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    return ExperimentConfig(model=model, data=data, trainer=trainer)


if __name__ == "__main__":
    train(config_from_args(parse_args()))

