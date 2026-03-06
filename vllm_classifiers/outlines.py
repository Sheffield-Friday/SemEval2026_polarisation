from typing import List, Optional

import outlines
import torch
from outlines import Generator
from outlines.types import Choice
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


class OutlinesClassifier:
    """Wrapper for Outlines to be compatible with OpenSetLLMClassifier."""

    def __init__(
        self,
        model_name: str,
        class_labels: List[str],
        conf_labels: Optional[List[int]] = None,
        device: str = "cuda",
        num_gpus: int = 1,
        max_model_len: Optional[int] = None,
    ):
        self.model_name = model_name
        self.class_labels = class_labels
        self.conf_labels = conf_labels
        self.device = device
        self.model = LLM(
            model=model_name, tensor_parallel_size=num_gpus, max_model_len=max_model_len
        )
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

        # set tokeniser and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # load model in outlines
        self.o_model = outlines.models.from_vllm_offline(self.model)

        # prebuild generator for performance
        self.pred_generator = Generator(
            self.o_model, output_type=Choice(self.class_labels)
        )
        if self.conf_labels:
            self.conf_generator = Generator(
                self.o_model, output_type=Choice(self.conf_labels)
            )
        else:
            self.conf_generator = None

    # use kwargs to fix compatibility with other interface
    def classify(
        self,
        messages: List[dict],
        verbal_confidence: bool = False,
        conf_prompt: Optional[str] = None,
        second_pred: bool = True,
        second_prompt: Optional[str] = None,
        **kwargs
    ) -> dict:
        torch.manual_seed(42)
        # TODO: clean this up
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pred = self.pred_generator(full_prompt, sampling_params=self.sampling_params)

        confidence = 1.0
        s_pred = None

        if verbal_confidence:
            if not self.conf_generator or not self.conf_labels:
                raise ValueError(
                    "conf_labels was not set when instantiating the classifier."
                )

            conf_messages = [
                {"role": "assistant", "content": str(pred)},
                {"role": "user", "content": conf_prompt},
            ]
            full_conf_prompt = self.tokenizer.apply_chat_template(
                messages + conf_messages, tokenize=False, add_generation_prompt=True
            )
            conf_str = str(
                self.conf_generator(
                    full_conf_prompt, sampling_params=self.sampling_params
                )
            ).strip()
            confidence = float(conf_str) / max(self.conf_labels)

            if second_pred and confidence != 1.0:
                if not second_prompt:
                    raise ValueError(
                        "When second_pred is set to True, second_prompt must have a value."
                    )
                s_pred_messages = [
                    {"role": "assistant", "content": conf_str},
                    {"role": "user", "content": second_prompt},
                ]
                s_pred_prompt = self.tokenizer.apply_chat_template(
                    messages + conf_messages + s_pred_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                s_pred = self.pred_generator(
                    s_pred_prompt, sampling_params=self.sampling_params
                )

        final_probs = {}
        pred_prob = 0.5 + (confidence / 2)
        min_conf = (1 - pred_prob) / (len(self.class_labels) - 1)

        if pred == s_pred:
            final_probs = {label: float(label == pred) for label in self.class_labels}
        else:
            for label in self.class_labels:
                assert label is not None
                if label == pred:
                    final_probs[label] = pred_prob
                elif label == s_pred:
                    final_probs[label] = 1 - pred_prob
                else:
                    if s_pred:
                        final_probs[label] = 0.0
                    else:
                        final_probs[label] = min_conf / 2

        # TODO: implement entropy and openset

        return {
            "predicted_label": pred,
            "confidence": confidence,
            "entropy": 0.0,
            "is_openset": False,
            "all_probs": final_probs,
            "alternative_label": None,
        }


class OutlinesMultiClassifier:
    """Wrapper for Outlines to be compatible with OpenSetLLMClassifier."""

    def __init__(
        self,
        model_name: str,
        class_labels: List[List[str]],
        conf_labels: Optional[List[int]] = None,
        device: str = "cuda",
        num_gpus: int = 1,
        max_model_len: Optional[int] = None,
    ):
        self.model_name = model_name
        self.class_labels = class_labels
        self.conf_labels = conf_labels
        self.device = device
        self.model = LLM(
            model=model_name, tensor_parallel_size=num_gpus, max_model_len=max_model_len
        )
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

        # set tokeniser and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # load model in outlines
        self.o_model = outlines.models.from_vllm_offline(self.model)

        # list of generators (one for each label)
        self.pred_generators = [
            Generator(self.o_model, output_type=Choice(c_labels))
            for c_labels in self.class_labels
        ]
        if self.conf_labels:
            self.conf_generator = Generator(
                self.o_model, output_type=Choice(self.conf_labels)
            )
        else:
            self.conf_generator = None

    # use kwargs to fix compatibility with other interface
    def classify(
        self,
        messages_list: List[List[dict]],
        **kwargs
    ) -> dict:
        torch.manual_seed(42)
        # TODO: clean this up
        if len(messages_list) != len(self.pred_generators):
            raise ValueError(
                "messages_list must be the same length as self.pred_generators"
            )

        full_prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_list
        ]

        pred = [
            pred_generator(full_prompts[i], sampling_params=self.sampling_params)
            for i, pred_generator in enumerate(self.pred_generators)
        ]

        return {
            "predicted_label": pred,
        }
