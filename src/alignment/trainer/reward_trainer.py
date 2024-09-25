from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
import trl


class RewardTrainer(trl.RewardTrainer):

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]
        chosen_probs = inputs["chosen_probs"]
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            losses = (
                -nn.functional.logsigmoid(rewards_chosen - rewards_rejected) * chosen_probs
                -nn.functional.logsigmoid(rewards_rejected - rewards_chosen) * (1.0 - chosen_probs)
            )
            loss = losses.mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
