import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, logging
from .projection import ProjectionHead
from .losses import TokenSupConLoss
from .base import SupConOutput, LinearProbeOutput

logging.set_verbosity_error()


class PhoBertSupCon(nn.Module):
    """
    Stage 1 Model: PhoBERT Backbone + Projection Head for Pure Contrastive Pre-training.
    Optimizes representation space via TokenSupConLoss (Google SupContrast 100% - NO ALPHA).
    Supports loading directly from a pre-trained model identifier (e.g. 'trungbb8/distilphobert')
    or from a pre-loaded RobertaConfig.
    """
    def __init__(self, model_name_or_config, projection_dim: int = 128, temperature: float = 0.07, o_label_id: int = 0):
        super(PhoBertSupCon, self).__init__()
        if isinstance(model_name_or_config, str):
            self.config = AutoConfig.from_pretrained(model_name_or_config)
            self.roberta = AutoModel.from_pretrained(model_name_or_config, config=self.config)
        else:
            self.config = model_name_or_config
            self.roberta = AutoModel.from_config(self.config)

        self.projection_head = ProjectionHead(
            input_dim=self.config.hidden_size,
            hidden_dim=256,
            output_dim=projection_dim
        )
        self.supcon_loss = TokenSupConLoss(temperature=temperature)
        self.o_label_id = o_label_id

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_masks=None):
        # 1. PhoBERT contextual representations
        outputs = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        seq_outputs = outputs[0]  # [B, max_len, hidden_dim]

        # 2. Subword alignment to original Vietnamese words via valid_ids
        batch_size = seq_outputs.size(0)
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_outputs.device).unsqueeze(1)
        word_representations = seq_outputs[range_vector, valid_ids]  # [B, max_words, 768]

        # 3. Filter valid entity tokens (exclude padding and background 'O')
        valid_entity_mask = (label_masks != 0) & (labels != self.o_label_id)
        num_valid = valid_entity_mask.sum().item()

        if num_valid < 2 or labels is None:
            return SupConOutput(loss=seq_outputs.new_tensor(0.0), num_valid_anchors=num_valid)

        # 4. Project entity token representations to hypersphere S^127
        entity_feats = self.projection_head(word_representations)[valid_entity_mask]
        entity_tags = labels[valid_entity_mask]

        # 5. Compute TokenSupConLoss (Vanilla Google SupContrast)
        loss = self.supcon_loss(entity_feats, entity_tags)
        return SupConOutput(loss=loss, num_valid_anchors=num_valid)


class PhoBertLinearProbe(nn.Module):
    """
    Stage 2 Model: Linear Probing Evaluation Protocol (Google SupContrast style).
    - Encoder Backbone: Loaded from Stage 1 checkpoint and FROZEN (torch.no_grad() / requires_grad=False).
    - Classifier: Single Linear layer (nn.Linear(768, num_labels)).
    - Objective: Standard Cross-Entropy Loss with ignore_index=-100.
    - Inference: Argmax per token (NO CRF, NO Viterbi decoding).
    """
    def __init__(self, model_name_or_config, num_labels: int = 11, freeze_backbone: bool = True):
        super(PhoBertLinearProbe, self).__init__()
        if isinstance(model_name_or_config, str):
            self.config = AutoConfig.from_pretrained(model_name_or_config)
            self.roberta = AutoModel.from_pretrained(model_name_or_config, config=self.config)
        else:
            self.config = model_name_or_config
            self.roberta = AutoModel.from_config(self.config)

        self.num_labels = num_labels
        self.freeze_backbone = freeze_backbone

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        if self.freeze_backbone:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freezes all parameters of the Transformer encoder."""
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Optional helper to unfreeze encoder for end-to-end fine-tuning."""
        for param in self.roberta.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_masks=None):
        batch_size = input_ids.size(0)
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=input_ids.device).unsqueeze(1)

        # 1. Forward through frozen backbone
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.roberta(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask
                )
                seq_outputs = outputs[0]
                word_representations = seq_outputs[range_vector, valid_ids]
        else:
            outputs = self.roberta(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            seq_outputs = outputs[0]
            word_representations = seq_outputs[range_vector, valid_ids]

        # 2. Linear classifier emission
        rep_dropped = self.dropout(word_representations)
        logits = self.classifier(rep_dropped)  # [B, max_words, num_labels]

        # 3. Predict tag sequences via Argmax (No CRF)
        active_mask = (label_masks != 0)
        preds = torch.argmax(logits, dim=-1)

        seq_tags = []
        for b in range(batch_size):
            length = active_mask[b].sum().item()
            seq_tags.append(preds[b, :length].cpu().tolist())

        # 4. Cross-Entropy Loss computation
        loss = None
        if labels is not None:
            active_loss = active_mask.view(-1)
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(-100, dtype=torch.long, device=labels.device)
            )
            loss = self.loss_fct(active_logits, active_labels)

        return LinearProbeOutput(loss=loss, logits=logits, tags=seq_tags)
