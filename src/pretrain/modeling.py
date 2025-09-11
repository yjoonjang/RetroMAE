import logging

import torch
from pretrain.arguments import ModelArguments
from pretrain.enhancedDecoder import ModernBertLayerForDecoder
from torch import nn
from transformers import ModernBertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

logger = logging.getLogger(__name__)


class RetroMAEForPretraining(nn.Module):
	def __init__(
			self,
			modernbert: ModernBertForMaskedLM,
			model_args: ModelArguments,
	):
		super(RetroMAEForPretraining, self).__init__()
		self.lm = modernbert

		self.decoder_embeddings = self.lm.model.embeddings
		self.c_head = ModernBertLayerForDecoder(modernbert.config)
		self.c_head.apply(self.lm._init_weights)

		self.cross_entropy = nn.CrossEntropyLoss()

		self.model_args = model_args

	def forward(self,
				encoder_input_ids, encoder_attention_mask, encoder_labels,
				decoder_input_ids, decoder_attention_mask, decoder_labels):
		lm_out: MaskedLMOutput = self.lm(
			encoder_input_ids, encoder_attention_mask,
			labels=encoder_labels,
			output_hidden_states=True,
			return_dict=True
		)
		cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D

		decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
		hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

		decoder_position_ids = self.lm.model.embeddings.position_ids[:, :decoder_input_ids.size(1)]
		decoder_position_embeddings = self.lm.model.embeddings.position_embeddings(decoder_position_ids)  # B L D
		query = decoder_position_embeddings + cls_hiddens

		matrix_attention_mask = self.lm.get_extended_attention_mask(
			decoder_attention_mask,
			decoder_attention_mask.shape,
			decoder_attention_mask.device
		)

		layer_output_tuple = self.c_head(
			query_states=query,
			key_value_states=hiddens,
			attention_mask=matrix_attention_mask,
			position_ids=decoder_position_ids,
		)
			
		hiddens = layer_output_tuple[0]
		pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

		return (loss + lm_out.loss,)

	def mlm_loss(self, hiddens, labels):
		pred_scores = self.lm.cls(hiddens)
		masked_lm_loss = self.cross_entropy(
			pred_scores.view(-1, self.lm.config.vocab_size),
			labels.view(-1)
		)
		return pred_scores, masked_lm_loss

	def save_pretrained(self, output_dir: str):
		self.lm.save_pretrained(output_dir)

	@classmethod
	def from_pretrained(
			cls, model_args: ModelArguments,
			*args, **kwargs
	):
		hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
		model = cls(hf_model, model_args)
		return model
