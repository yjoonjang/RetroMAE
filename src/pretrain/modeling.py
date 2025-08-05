import logging
from copy import deepcopy
from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding

import torch
from pretrain.arguments import ModelArguments
from pretrain.enhancedDecoder import BertLayerForDecoder
from torch import nn
from transformers import ModernBertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from utils import unpad_input, pad_input

logger = logging.getLogger(__name__)

class OneStepDecoder(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.layer = base_layer            # ModernBertEncoderLayer(eager+padded RoPE)

    def forward(self, query, key, value,
                attention_mask=None,
                position_ids=None,         # ★ 받도록 선언
                **kw):
        hidden_states = query              # RetroMAE: Q = CLS-broadcast
        if position_ids is None:
            # (B,L) position_ids = 0..L-1
            B, L, _ = hidden_states.size()
            position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, L)

        return self.layer(hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,   # ★ 넘겨줌
            **kw
        )

class RetroMAEForPretraining(nn.Module):
    def __init__(
            self,
            modernbert: ModernBertForMaskedLM,
            model_args: ModelArguments,
    ):
        super(RetroMAEForPretraining, self).__init__()
        self.lm = modernbert

        self.decoder_embeddings = self.lm.model.embeddings
        # self.c_head = BertLayerForDecoder(modernbert.config)
        decoder_layer = deepcopy(self.lm.model.layers[-1])
        decoder_layer.config._attn_implementation = "eager" 
        
        rot_cfg = deepcopy(modernbert.config)
        rot_cfg.rope_scaling = None           
        rot_cfg.rope_type = "default"      
        rot_cfg.base = 160000.0       
        decoder_layer.attn.rotary_emb = ModernBertRotaryEmbedding(rot_cfg)
        self.c_head = OneStepDecoder(decoder_layer)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):
        # return (torch.sum(self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]), )
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, 
            encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        # cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D
        last_hidden = lm_out.hidden_states[-1]              # (T,H) or (B,L,H)
        if last_hidden.dim() == 2:                          # unpadded → repad
            B, L = encoder_attention_mask.shape
            H = last_hidden.size(-1)
            repad = torch.zeros(B * L, H,
                                device=last_hidden.device,
                                dtype=last_hidden.dtype)
            repad[encoder_attention_mask.view(-1).bool()] = last_hidden
            last_hidden = repad.view(B, L, H)

        cls_hiddens = last_hidden[:, :1]

        # cls_hiddens = padded_hidden[:, :1]

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        # tok = self.decoder_embeddings.tok_embeddings(decoder_input_ids)
        # tok = self.decoder_embeddings.norm(tok)
        # decoder_embedding_output = self.decoder_embeddings.drop(tok) 
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        # decoder_position_ids = self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]
        # decoder_position_embeddings = self.lm.bert.embeddings.position_embeddings(decoder_position_ids)  # B L D
        # query = decoder_position_embeddings + cls_hiddens

        seq_len = hiddens.size(1)
        query = cls_hiddens.expand(-1, seq_len, -1).contiguous() 

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return (loss + lm_out.loss,)

    def mlm_loss(self, hiddens, labels):
        hidden = self.lm.head(hiddens)
        pred_scores = self.lm.decoder(hidden)
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
