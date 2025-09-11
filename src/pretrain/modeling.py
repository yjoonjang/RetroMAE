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
        self.layer = base_layer            

    def forward(self, query, key, value,
                attention_mask=None,
                position_ids=None,         
                **kw):
        hidden_states = query              
        if position_ids is None:
            B, L, _ = hidden_states.size()
            position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, L)

        return self.layer(hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
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

        # 1. 인코더에서 hs
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids,
            encoder_attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True
        )
        last_hidden = lm_out.hidden_states[-1]

        if last_hidden.dim() == 2:
            B, L = encoder_attention_mask.shape
            H = last_hidden.size(-1)
            # 0으로 채워진 전체 크기의 텐서를 만들고
            repad = torch.zeros(B * L, H, device=last_hidden.device, dtype=last_hidden.dtype)
            # 실제 토큰이 있는 위치에만 hidden_state를 채워 넣습니다.
            repad[encoder_attention_mask.view(-1).bool()] = last_hidden
            # 마지막으로 [B, L, H] 형태로 재구성합니다.
            last_hidden = repad.view(B, L, H)

        # 2. 마스킹된 토큰 위치 & 그 토큰들의 hs
        masked_indices = (encoder_labels != -100)
        masked_hidden_states = last_hidden[masked_indices]
        
        prediction_head_output = self.lm.head(masked_hidden_states)
        masked_logits = self.lm.decoder(prediction_head_output)

        # 3. 마스킹된 위치의 정답 레이블
        masked_true_labels = encoder_labels[masked_indices]
        loss_enc = self.cross_entropy(masked_logits, masked_true_labels)

        # 4. decoder 부분
        cls_hiddens = last_hidden[:, :1]

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

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

        pred_scores, loss_dec = self.mlm_loss(hiddens, decoder_labels)

        # 5. 두 loss 합쳐서 반환
        return (loss_enc + loss_dec,)

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
