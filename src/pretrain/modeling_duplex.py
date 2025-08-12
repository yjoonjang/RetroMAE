import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForMaskedLM, ModernBertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding

from pretrain.arguments import ModelArguments
from pretrain.modeling import OneStepDecoder

logger = logging.getLogger(__name__)


class DupMAEForPretraining(nn.Module):
    def __init__(
            self,
            modernbert: ModernBertForMaskedLM,
            model_args: ModelArguments,
    ):
        super(DupMAEForPretraining, self).__init__()
        self.lm = modernbert

        # Decoder setup (RetroMAE와 동일)
        self.decoder_embeddings = self.lm.model.embeddings
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

    def ot_embedding(self, logits, attention_mask):
        """Bag-of-Words를 위한 임베딩 생성"""
        # 패딩된 부분은 -inf로 마스킹하여 max 연산에서 무시되도록 함
        mask = (1 - attention_mask.unsqueeze(-1)) * -1000.0
        reps, _ = torch.max(logits + mask, dim=1)  # [B, L, V] -> [B, V]
        return reps

    def decoder_ot_loss(self, ot_embedding, bag_word_weight):
        """Bag-of-Words 손실 계산"""
        log_probs = F.log_softmax(ot_embedding, dim=-1)
        bow_loss = -torch.sum(bag_word_weight * log_probs, dim=1)
        return bow_loss.mean()

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels,
                bag_word_weight):

        # 1. 인코더에서 hidden_state만 얻음 (labels 전달 X)
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids,
            encoder_attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True
        )
        last_hidden = lm_out.hidden_states[-1]

        # 2. 2D 출력을 3D로 복원 (A.X-Encoder 호환성)
        if last_hidden.dim() == 2:
            B, L = encoder_attention_mask.shape
            H = last_hidden.size(-1)
            repad = torch.zeros(B * L, H, device=last_hidden.device, dtype=last_hidden.dtype)
            repad[encoder_attention_mask.view(-1).bool()] = last_hidden
            last_hidden = repad.view(B, L, H)

        # --- 3. 인코더 MLM 손실 (Sparse) ---
        masked_indices_enc = (encoder_labels != -100)
        masked_hidden_states_enc = last_hidden[masked_indices_enc]
        
        prediction_head_output_enc = self.lm.head(masked_hidden_states_enc)
        masked_logits_enc = self.lm.decoder(prediction_head_output_enc)

        masked_true_labels_enc = encoder_labels[masked_indices_enc]
        loss_enc = self.cross_entropy(masked_logits_enc, masked_true_labels_enc)

        # --- 4. Bag-of-Words 손실 (Dense) ---
        # bow_loss를 위해 전체 hidden_state로 dense logits을 생성
        dense_prediction_head_output = self.lm.head(last_hidden)
        dense_logits = self.lm.decoder(dense_prediction_head_output)
        
        # [CLS] 토큰 제외하고 ot_embedding 계산
        ot_embeds = self.ot_embedding(dense_logits[:, 1:], encoder_attention_mask[:, 1:])
        bow_loss = self.decoder_ot_loss(ot_embeds, bag_word_weight=bag_word_weight)

        # --- 5. 디코더 MLM 손실 (Sparse) ---
        cls_hiddens = last_hidden[:, :1]
        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens_dec = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        seq_len = hiddens_dec.size(1)
        query = cls_hiddens.expand(-1, seq_len, -1).contiguous()

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens_dec = self.c_head(query=query,
                                  key=hiddens_dec,
                                  value=hiddens_dec,
                                  attention_mask=matrix_attention_mask)[0]

        _, loss_dec = self.mlm_loss(hiddens_dec, decoder_labels)

        # --- 6. 최종 손실 결합 ---
        total_loss = loss_enc + loss_dec + self.model_args.bow_loss_weight * bow_loss
        return (total_loss,)

    def mlm_loss(self, hiddens, labels):
        masked_indices = (labels != -100)
        masked_hidden_states = hiddens[masked_indices]

        prediction_head_output = self.lm.head(masked_hidden_states)
        masked_logits = self.lm.decoder(prediction_head_output)

        masked_true_labels = labels[masked_indices]
        masked_lm_loss = self.cross_entropy(masked_logits, masked_true_labels)

        return None, masked_lm_loss

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
