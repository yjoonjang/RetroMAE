import logging

import torch
import torch.nn.functional as F
from pretrain.arguments import ModelArguments
from pretrain.enhancedDecoder import ModernBertLayerForDecoder
from torch import nn
from transformers import ModernBertForMaskedLM, AutoModelForMaskedLM  
from transformers.modeling_outputs import MaskedLMOutput

logger = logging.getLogger(__name__)


class DupMAEForPretraining(nn.Module):
	def __init__(
			self,
			modernbert: ModernBertForMaskedLM,  
			model_args: ModelArguments,
	):
		super(DupMAEForPretraining, self).__init__()
		self.lm = modernbert  

		self.decoder_embeddings = self.lm.model.embeddings  
		self.c_head = ModernBertLayerForDecoder(modernbert.config) 
		self.c_head.apply(self.lm._init_weights)

		self.cross_entropy = nn.CrossEntropyLoss()

		self.model_args = model_args

	def decoder_mlm_loss(self, sentence_embedding, decoder_input_ids, decoder_attention_mask, decoder_labels):
		B, L = decoder_input_ids.size(0), decoder_input_ids.size(1)
		device = decoder_input_ids.device
		
		if sentence_embedding.dim() == 2:
			sentence_embedding = sentence_embedding.unsqueeze(1)  # [B, D] -> [B, 1, D]

		decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
		hiddens = torch.cat([sentence_embedding, decoder_embedding_output[:, 1:]], dim=1)

		query_embedding = sentence_embedding.expand(B, L, sentence_embedding.size(-1))    # [B, L, H]
		query = self.decoder_embeddings(inputs_embeds=query_embedding)                    # [B, L, H]

		position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

		matrix_attention_mask = self.lm.get_extended_attention_mask(
			decoder_attention_mask,
			decoder_attention_mask.shape,
			decoder_attention_mask.device
		)

		layer_output = self.c_head(
			query_states=query,
			key_value_states=hiddens,
			attention_mask=matrix_attention_mask,
			position_ids=position_ids
		)
		
		hiddens = layer_output[0]
		pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

		return loss

	def ot_embedding(self, logits, attention_mask):
		mask = (1 - attention_mask.unsqueeze(-1)) * -1000
		reps, _ = torch.max(logits + mask, dim=1)  # B V
		return reps

	def decoder_ot_loss(self, ot_embedding, bag_word_weight):
		input = F.log_softmax(ot_embedding, dim=-1)
		bow_loss = torch.mean(-torch.sum(bag_word_weight * input, dim=1))
		return bow_loss

	def forward(self,
				encoder_input_ids, encoder_attention_mask, encoder_labels,
				decoder_input_ids, decoder_attention_mask, decoder_labels,
				bag_word_weight):
		lm_out: MaskedLMOutput = self.lm(
			encoder_input_ids, encoder_attention_mask,
			labels=encoder_labels,
			output_hidden_states=True,
			return_dict=True
		)

		# # RetroMAE와 동일한 방식으로 CLS 추출
		# seqlens_in_batch = encoder_attention_mask.sum(dim=-1, dtype=torch.int32)
		# cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
		# cls_indices = cu_seqlens[:-1]
		# print(lm_out.hidden_states[-1].shape)
		# unpadded_hiddens = lm_out.hidden_states[-1]
		# cls_hiddens = unpadded_hiddens[cls_indices]  # [batch_size, hidden_dim] 형태가 됨
		last_hidden = lm_out.hidden_states[-1]
		if last_hidden.dim() == 3:
			cls_hiddens = last_hidden[:, 0]
		else:
			seqlens = encoder_attention_mask.sum(dim=-1, dtype=torch.int32)           
			cu = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
			cls_indices = cu[:-1]                                                    
			cls_hiddens = last_hidden[cls_indices]                                       


		decoder_mlm_loss = self.decoder_mlm_loss(cls_hiddens, decoder_input_ids, decoder_attention_mask, decoder_labels)

		ot_embedding = self.ot_embedding(lm_out.logits[:, 1:], encoder_attention_mask[:, 1:])
		bow_loss = self.decoder_ot_loss(ot_embedding, bag_word_weight=bag_word_weight)

		total_loss = decoder_mlm_loss + self.model_args.bow_loss_weight * bow_loss + lm_out.loss

		# wandb 로깅
		return {
			'loss': total_loss,
			'encoder_mlm_loss': lm_out.loss,
			'decoder_mlm_loss': decoder_mlm_loss,
			'bow_loss': bow_loss,
			# 'weighted_bow_loss': self.model_args.bow_loss_weight * bow_loss
		}

	def mlm_loss(self, hiddens, labels):
		# RetroMAE와 동일한 방식으로 변경
		head_output = self.lm.head(hiddens)
		pred_scores = self.lm.decoder(head_output)
		
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
