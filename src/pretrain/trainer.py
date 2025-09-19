import logging
import os
from typing import Dict, Optional

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class PreTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        logs["step"] = self.state.global_step
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        
        if isinstance(outputs, dict):
            loss = outputs['loss']
            # wandb에 각 loss 로깅
            if self.state.global_step > 0:  # 첫 번째 step 이후에 로깅
                self.log({
                    'train/encoder_mlm_loss': outputs['encoder_mlm_loss'].item(),
                    'train/decoder_mlm_loss': outputs['decoder_mlm_loss'].item(), 
                    'train/bow_loss': outputs['bow_loss'].item(),
                    # 'train/weighted_bow_loss': outputs['weighted_bow_loss'].item(),
                })
        else:
            # RetroMAE의 경우
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            
        return (loss, outputs) if return_outputs else loss
