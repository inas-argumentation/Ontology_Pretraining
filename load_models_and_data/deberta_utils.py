from typing import Any, Optional, Union, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.deberta.modeling_deberta import (
    DebertaPreTrainedModel,
    DebertaModel,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
)
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss


class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.lm_predictions = DebertaOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        pass

    def set_output_embeddings(self, new_embeddings):
        pass

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_predictions(sequence_output, self.deberta.embeddings.word_embeddings)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaLMPredictionHead(nn.Module):
    """https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py#L270"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    # note that the input embeddings must be passed as an argument
    def forward(self, hidden_states, word_embeddings):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(
            hidden_states)  # original used MaskedLayerNorm, but passed no mask. This is equivalent.
        hidden_states = torch.matmul(hidden_states, word_embeddings.weight.t()) + self.bias
        return hidden_states


class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = DebertaLMPredictionHead(config)

    # note that the input embeddings must be passed as an argument
    def forward(self, sequence_output, word_embeddings):
        prediction_scores = self.lm_head(sequence_output, word_embeddings)
        return prediction_scores


#### V2


class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.lm_predictions = DebertaV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        # TODO: implement
        pass

    def set_output_embeddings(self, new_embeddings):
        # TODO: implement
        pass

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_predictions(sequence_output, self.deberta.embeddings.word_embeddings)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV2LMPredictionHead(nn.Module):
    """https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py#L270"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    # note that the input embeddings must be passed as an argument
    def forward(self, hidden_states, word_embeddings):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(
            hidden_states)  # original used MaskedLayerNorm, but passed no mask. This is equivalent.
        hidden_states = torch.matmul(hidden_states, word_embeddings.weight.t()) + self.bias
        return hidden_states


# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = DebertaV2LMPredictionHead(config)

    # note that the input embeddings must be passed as an argument
    def forward(self, sequence_output, word_embeddings):
        prediction_scores = self.lm_head(sequence_output, word_embeddings)
        return prediction_scores