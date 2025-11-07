import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, PeftModel, get_peft_model


class TextEmbeddingModel(nn.Module):
    """Wrapper around a Hugging Face model with optional LoRA adapters."""

    def __init__(
        self,
        model_name,
        output_hidden_states=False,
        lora=False,
        infer=False,
        use_pooling="average",
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0,
        adapter_path=None,
    ):
        super(TextEmbeddingModel, self).__init__()
        self.model_name = model_name
        self.use_pooling = use_pooling
        self.lora = lora
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {"trust_remote_code": True}
        if output_hidden_states:
            model_kwargs["output_hidden_states"] = True
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)

        if self.lora:
            peft_config = LoraConfig(
                peft_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=infer,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)
            if adapter_path is not None:
                self.load_adapter(adapter_path, is_trainable=not infer)
            else:
                self.model.print_trainable_parameters()
        elif adapter_path is not None:
            self.model = AutoModel.from_pretrained(adapter_path, **model_kwargs)

    def pooling(self, model_output, attention_mask, hidden_states=False):
        if hidden_states:
            if self.use_pooling == "average":
                model_output.masked_fill(~attention_mask[None, ..., None].bool(), 0.0)
                emb = model_output.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            elif self.use_pooling == "max":
                emb = model_output.masked_fill(~attention_mask[None, ..., None].bool(), float("-inf"))
                emb, _ = emb.max(dim=2)
            elif self.use_pooling == "cls":
                emb = model_output[:, :, 0]
            else:
                raise ValueError("Pooling method not supported")
            emb = emb.permute(1, 0, 2)
        else:
            if self.use_pooling == "average":
                model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
                emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            elif self.use_pooling == "max":
                emb = model_output.masked_fill(~attention_mask[..., None].bool(), float("-inf"))
                emb, _ = emb.max(dim=1)
            elif self.use_pooling == "cls":
                emb = model_output[:, 0]
            else:
                raise ValueError("Pooling method not supported")
        return emb

    def forward(self, encoded_batch, hidden_states=False, retrun_all_emb=False):
        if "t5" in self.model_name.lower():
            input_ids = encoded_batch['input_ids']
            decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
            model_output = self.model(**encoded_batch,
                                  decoder_input_ids=decoder_input_ids)
        else:
            model_output = self.model(**encoded_batch)
        
        
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            if hidden_states:
                model_output = model_output["hidden_states"]
                model_output = torch.stack(model_output, dim=0)
            else:
                model_output = model_output["last_hidden_state"]

        emb = self.pooling(model_output, encoded_batch['attention_mask'], hidden_states)
        if retrun_all_emb:
            return emb, model_output
        return emb

    def save_pretrained(self, save_directory: str, save_tokenizer: bool = True):
        os.makedirs(save_directory, exist_ok=True)
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_directory)
        else:
            self.model.save_pretrained(save_directory)
        if save_tokenizer:
            self.tokenizer.save_pretrained(save_directory)

    def load_adapter(self, adapter_path: str, is_trainable: bool = False):
        if not self.lora or not isinstance(self.model, PeftModel):
            raise ValueError("LoRA is not enabled for this model instance.")
        self.model = PeftModel.from_pretrained(
            self.model.base_model,
            adapter_path,
            is_trainable=is_trainable,
        )
        self.model.print_trainable_parameters()

    def merge_and_unload(self):
        if not isinstance(self.model, PeftModel):
            raise ValueError("The current model does not contain a LoRA adapter to merge.")
        merged_model = self.model.merge_and_unload()
        return merged_model


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size,num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class TextClassificationModel(nn.Module):
    def __init__(self, opt,dim=2):
        super(TextClassificationModel, self).__init__()
        self.model = TextEmbeddingModel(opt.model_name,lora=True,use_pooling=opt.pooling,\
                                        lora_r=opt.lora_r,lora_alpha=opt.lora_alpha,infer=True)
        self.root_classfier = nn.Linear(opt.embedding_dim, dim)

    def forward(self, encoded_batch):
        q = self.model(encoded_batch)
        out = self.root_classfier(q)
        return out

    