import logging
import math
import os
import random
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DPOConfig,
    DPOTrainer,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel


logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    os.environ["WANDB_PROJECT"] = training_args.run_name
    training_args.save_only_model = True

    #######
    # Setup
    #######
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
    if training_args.filter_threshold and training_args.filter_threshold < 0.0:
        training_args.filter_threshold = None
    run_name = "-".join([
        f"b{batch_size}",
        f"lr{training_args.learning_rate}",
        f"e{training_args.num_train_epochs}",
        f"b{training_args.beta}",
        f"btb{training_args.bt_beta or 'inf'}",
        f"ft{training_args.filter_threshold if training_args.filter_threshold is not None else 'n'}",
        f"seed{training_args.seed}",
    ])
    if "wandb" in training_args.report_to:
        training_args.tracker_kwargs = {"wandb": {"name": run_name}}
    training_args.run_name = run_name
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.run_name,
    )

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    print(raw_datasets)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    def maybe_insert_system_message(messages, tokenizer):
        if messages[0]["role"] == "system":
            return
        # chat template can be one of two attributes, we check in order
        chat_template = tokenizer.chat_template
        if chat_template is None:
            chat_template = tokenizer.default_chat_template
        # confirm the jinja template refers to a system message before inserting
        if "system" in chat_template or "<|im_start|>" in chat_template:
            messages.insert(0, {"role": "system", "content": ""})

    def preprocess_shp(example, tokenizer, auto_insert_empty_system_msg=True):
        prompt_messages = [
            {"role": "user", "content": example['history']},
        ]
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(prompt_messages, tokenizer)

        if example['score_A'] > example['score_B']:
            score_chosen = example['score_A']
            score_rejected = example['score_B']
            chosen = example['human_ref_A']
            rejected = example['human_ref_B']
        else:
            score_chosen = example['score_B']
            score_rejected = example['score_A']
            chosen = example['human_ref_B']
            rejected = example['human_ref_A']

        chosen_messages = [
            {"role": "assistant", "content": chosen},
        ]
        rejected_messages = [
            {"role": "assistant", "content": rejected},
        ]
        example['text_prompt'] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        example['text_chosen'] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        example['text_rejected'] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        example['score_chosen'] = score_chosen
        example['score_rejected'] = score_rejected
        return example

    raw_datasets = raw_datasets.map(
        preprocess_shp,
        fn_kwargs={
            "tokenizer": tokenizer,
            "auto_insert_empty_system_msg": True,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    soft_label_dict = {}
    if training_args.soft_label_json:
        import json
        with open(training_args.soft_label_json, "r") as f:
            for d in json.load(f):
                soft_label_dict[d["prompt_id"]] = d

    def preprocess_function(example):
        prompt_id = example.get("prompt_id", None)
        chosen_text, rejected_text = example["text_chosen"], example["text_rejected"]
        if "score_chosen" in example and "score_rejected" in example:
            score_chosen, score_rejected = example["score_chosen"], example["score_rejected"]
        elif "chosen_rating" in example and "rejected_rating" in example:
            score_chosen, score_rejected = example["chosen_rating"], example["rejected_rating"]

        # Assume p follows the Bradley-Terry model.
        p_chosen = 1.0
        if training_args.bt_beta:
            score_diff = score_chosen - score_rejected
            p_chosen = 1.0 / (1.0 + math.exp(-training_args.bt_beta * score_diff))
        if soft_label_dict and prompt_id in soft_label_dict:
            p_chosen = soft_label_dict[prompt_id]["p_chosen"]
            if p_chosen < 0.5:
                p_chosen = 1.0 - p_chosen
                chosen_text, rejected_text = rejected_text, chosen_text

        example["text_chosen"] = chosen_text
        example["text_rejected"] = rejected_text
        example["p_chosen"] = p_chosen
        return example

    raw_datasets = raw_datasets.map(
        preprocess_function,
        num_proc=data_args.preprocessing_num_workers,
    )

    def prevent_overfit(dataset):  # following kto
        all_prompts = dict()
        for i, d in enumerate(dataset):
            p = d['text_prompt'].strip()
            if p in all_prompts:
                all_prompts[p].append(i)
            else:
                all_prompts[p] = [i]

        exclude_idx = []
        for k in all_prompts.keys():
            if len(all_prompts[k]) > 5:
                to_rm = all_prompts[k][5:]
                exclude_idx.extend(to_rm)

        origin_len = len(dataset)
        dataset = dataset.select(
            (
                i for i in range(len(dataset)) if i not in set(exclude_idx)
            )
        )
        print(f"## {origin_len - len(dataset)} is excluded")
        print(f"## {origin_len} --> {len(dataset)}")
        return dataset

    for split in ["train", "test"]:
        raw_datasets[split] = prevent_overfit(raw_datasets[split])

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
    print(raw_datasets)
    print(raw_datasets['train'][0])

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map(),
        quantization_config=get_quantization_config(model_args),
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        # load the model, merge the adapter weights and unload the adapter
        # Note: to run QLora, you will need to merge the based model separately as the merged model in 16bit
        logger.info(f"Merging peft adapters for {model_args.model_name_or_path=}")

        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)

        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model, model_args.model_name_or_path, revision=model_args.model_revision
        )
        model.eval()
        model = model.merge_and_unload()
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        loss_type=training_args.loss_type,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
    )
    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
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
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
