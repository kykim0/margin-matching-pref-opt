# Reward modeling on preference data.

import logging
import math
import os
import sys

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    RewardDataCollatorWithPadding,
    RewardTrainer,
    RMConfig,
    apply_chat_template,
    compute_accuracy,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
import torch
import transformers
from transformers import AutoModelForSequenceClassification, set_seed

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, RMConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    num_gpus = torch.cuda.device_count()
    per_device_batch = training_args.per_device_train_batch_size
    grad_accum_steps = training_args.gradient_accumulation_steps
    batch_size = num_gpus * per_device_batch * grad_accum_steps
    run_name = "-".join([
        f"b{batch_size}",
        f"lr{training_args.learning_rate}",
        f"s{max(0, training_args.max_steps)}",
        f"e{training_args.num_train_epochs}",
        f"btb{training_args.bt_beta or 'inf'}",
        f"seed{training_args.seed}",
    ])
    if "wandb" in training_args.report_to:
        training_args.tracker_kwargs = {"wandb": {"name": run_name}}
    training_args.run_name = run_name
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.run_name,
    )

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["chosen", "rejected", "prompt", "score_chosen", "score_rejected", "source"],
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(fn for fn in raw_datasets["train"].features if fn not in ["score_chosen", "score_rejected", "source"])

    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.padding_side = "right"

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "rm",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "chosen_probs": [],
        }
        for (chosen_text, rejected_text, score_chosen, score_rejected) in zip(
            examples["text_chosen"], examples["text_rejected"],
            examples["score_chosen"], examples["score_rejected"]):
            # Assume p follows the Bradley-Terry model.
            chosen_p = 1.0
            if training_args.bt_beta:
                score_diff = score_chosen - score_rejected
                chosen_p = 1.0 / (1.0 + math.exp(-training_args.bt_beta * score_diff))
            new_examples["chosen_probs"].append(chosen_p)

            # Ensure consistency with how the model is to be invoked, e.g., during PPO.
            chosen_tokenized = tokenizer(chosen_text)
            rejected_tokenized = tokenizer(rejected_text)

            new_examples["input_ids_chosen"].append(chosen_tokenized["input_ids"])
            new_examples["attention_mask_chosen"].append(chosen_tokenized["attention_mask"])
            new_examples["input_ids_rejected"].append(rejected_tokenized["input_ids"])
            new_examples["attention_mask_rejected"].append(rejected_tokenized["attention_mask"])

        return new_examples

    # Preprocess and filter examples that are longer than max_length.
    def length_filter(x):
        return (len(x["input_ids_chosen"]) <= training_args.max_length and
                len(x["input_ids_rejected"]) <= training_args.max_length)

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )
    raw_datasets = raw_datasets.filter(length_filter)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        **model_kwargs,
    )
    logger.info("*** Model loaded! ***")
    if getattr(model, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    #########################
    # Instantiate the Trainer
    #########################
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        data_collator=RewardDataCollatorWithPadding(tokenizer, max_length=training_args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
        peft_config=get_peft_config(model_args),
    )
    trainer.train(training_args.resume_from_checkpoint)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process.
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
