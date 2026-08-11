#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from utils import CodeDataset, get_last_checkpoint
from sklearn.metrics import recall_score, precision_score, f1_score
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from modeling_roberta import RobertaForSequenceClassification, RobertaConfig
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from petl.options import (TuneArguments)
from petl.petl_enc_model import PETLEncModel

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    max_tokens_per_batch: Optional[int] = field(
        default=0,
        metadata={
            "help": "dynamic batching. Override batch size when larger than 0"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    train_filename: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    valid_filename: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_filename: Optional[str] = field(default=None,
                                         metadata={"help": "A csv or a json file containing the test data."})
    data_num: Optional[int] = field(
        default=-1,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               TuneArguments))
    model_args, data_args, training_args, tune_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    config = RobertaConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2
    )
    if training_args.do_predict and not training_args.do_train:
        config = RobertaConfig.from_pretrained(
            last_checkpoint,
            num_labels=2
        )

    # put useful args into config: these arguments will be used in models, thus adding them to config
    # interested_args = ['use_prefix', 'mid_dim', 'preseqlen', 'prefix_dropout', 'unfreeze_params']
    for k, v in vars(tune_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)

    setattr(training_args, 'max_tokens_per_batch', data_args.max_tokens_per_batch)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    dataset_loader = CodeDataset(os.path.dirname(model_args.train_filename))

    label_list = [0, 1]
    label_to_id = {v: i for i, v in enumerate(label_list)}

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = ((examples['code1'], examples['code2']))

        # {input_ids, attention_mask}
        result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation='longest_first')
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    if tune_args.attn_mode != "none" or tune_args.ffn_mode != "none":
        model = PETLEncModel(config, tune_args, model)
        model.print_trainable_parameters()

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        eval_acc = np.mean(p.label_ids == preds)
        recall = recall_score(p.label_ids, preds, average='macro')
        precision = precision_score(p.label_ids, preds, average='macro')
        f1 = f1_score(p.label_ids, preds, average='macro')
        results = {
            "accuracy": round(eval_acc, 4),
            "f1": round(f1, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
        }
        return results

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Training
    if training_args.do_train:
        train_dataset = dataset_loader.load(model_args.train_filename, model_args.data_num)
        eval_dataset = dataset_loader.load(model_args.valid_filename, model_args.data_num)
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on train dataset",
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on eval dataset",
        )
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_predict:
        test_dataset = dataset_loader.load(model_args.test_filename, model_args.data_num)
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on test dataset",
        )
        if not training_args.do_train:

            logger.info("load model state from %s", last_checkpoint)
            model.load_state_dict(torch.load(os.path.join(last_checkpoint, "pytorch_model.bin")))
            # Initialize our Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=None,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
        metrics = trainer.evaluate(metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
