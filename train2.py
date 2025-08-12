import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import AutoModel, AutoTokenizer
from transformers.integrations import deepspeed
import pdb


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import random
import pandas as pd
from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, BitsAndBytesConfig,  TrainingArguments, Trainer
from transformers import (
    Idefics2ForConditionalGeneration, 
    Idefics2Model, 
    Idefics2Config,
)
from transformers.trainer import *

from transformers.models.idefics2.modeling_idefics2 import (
    Idefics2VisionTransformer,
    ModelOutput,
    Idefics2Connector,
    logger,DynamicCache,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    
    Idefics2BaseModelOutputWithPast,
    Cache, DynamicCache,
    IDEFICS2_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
)
from prompt_pool import PromptPool
import pickle
import io

import datetime
import argparse
import numpy as np
from einops import rearrange


idefics_path="HuggingFaceM4/idefics2-8b"





def set_all_seed(seed):
   
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 


set_all_seed(42)



local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}

def create_arg_parser():
    """
    Create an ArgumentParser object to parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="debug",
    )
   
    parser.add_argument(
        "--lora_path",
        type=str,
        default="ref8_wofp16_newloss_1pmpt100_acc4trbz2_10.14_11.39.30/checkpoint-5000",
    )
    parser.add_argument(
        "--ref_pair_num",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--grad_acc_step",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--train_bz",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--pull_constraint_coeff",
        type=float,
        default=1,
    )

    
    parser.add_argument(
        "--length",
        type=int,
        default=10,
    )  
    
    parser.add_argument(
        "--fp16",
        type=int,
        default=1,
    )  
    parser.add_argument(
        "--eval_bz",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2,
    )
    
    parser.add_argument(
        "--use_prompt",
        type=int,
        default=0,
    )   


    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=77,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default='ds_config.json',
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )
    
    args=parser.parse_args()
    return args


@dataclass

class Idefics2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Idefics2 causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
     
    reduce_sim: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        def process_inputs(inputs, idx):
            
            sub_inputs = {key: val[:,idx:idx+1,:].contiguous().reshape(val.shape[0], *val.shape[2:]) for key, val in inputs.items()}
            targets = sub_inputs.pop("targets").squeeze(1)
            res = model(**sub_inputs) 
            
            logits = res.get("logits")
            
            reduce_sim = res.get("reduce_sim")
            
            return logits, targets, reduce_sim 
         
        negative_logits, negative_targets, reduce_sim_n = process_inputs(inputs, 0)
        positive_logits, positive_targets, reduce_sim_p = process_inputs(inputs, 1)

        
        
        
        loss_fct = torch.nn.CrossEntropyLoss().to(DEVICE)
        loss = (loss_fct(negative_logits[:, -1], negative_targets) + 
                loss_fct(positive_logits[:, -1], positive_targets)) / 2
        
       
        pos_rel_logits_p = positive_logits[:, -1][:, 648]
        neg_rel_logits_p = negative_logits[:, -1][:, 648]
        
        rel_logits_p = torch.stack([ pos_rel_logits_p, neg_rel_logits_p], dim=-1)
        
    
        rel_labels_p = torch.zeros(pos_rel_logits_p.shape[0], device=model.device, dtype=torch.long)
        rel_loss_p = torch.nn.functional.cross_entropy(rel_logits_p, rel_labels_p)


        
        pos_rel_logits_n = positive_logits[:, -1][:, 387]
        neg_rel_logits_n = negative_logits[:, -1][:, 387]
        rel_logits_n = torch.stack([ pos_rel_logits_n, neg_rel_logits_n], dim=-1)
        
        
        rel_labels_n = torch.ones(pos_rel_logits_p.shape[0], device=model.device, dtype=torch.long) 
        rel_loss_n = torch.nn.functional.cross_entropy(rel_logits_n, rel_labels_n)
        

        loss = loss + rel_loss_p + rel_loss_n
        
       
        try:
            pull_constraint_coeff=model.module.model.prompt_pool.pull_constraint_coeff
        except:
            pull_constraint_coeff=model.model.prompt_pool.pull_constraint_coeff
                     
        if pull_constraint_coeff!=0:
            
            

            loss = loss - pull_constraint_coeff * (reduce_sim_n + reduce_sim_p)
            
        torch.cuda.empty_cache()
        return (loss, negative_logits) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        
        
        
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            
            try:
                torch.save(self.model.model.prompt_pool.state_dict(), os.path.join(output_dir, "prompt_pool_state_dict.pt"))

            except:
                torch.save(self.model.model.prompt_pool,os.path.join(output_dir,"prompt_pool.pt"))
            
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_xpu_available():
                torch.xpu.empty_cache()
            elif is_mlu_available():
                torch.mlu.empty_cache()
            elif is_npu_available():
                torch.npu.empty_cache()
            elif is_torch_version(">=", "2.0") and is_mps_available():
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean() 
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            
           
            self.accelerator.backward(loss, **kwargs)
            

            
            
        return loss.detach() / self.args.gradient_accumulation_steps

    def _save_checkpoint(self, model, trial, metrics=None):
        
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            
            self._save_optimizer_and_scheduler(output_dir)
           
            self._save_rng_state(output_dir)

       
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

       
        if self.args.should_save:
            
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

       
        if self.args.should_save:
            
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)


class Idefics2ModelPreferTokens(Idefics2Model):
    def __init__(self, config: Idefics2Config, **model_kwargs):
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = Idefics2VisionTransformer(config.vision_config)
        self.connector = Idefics2Connector(config)
       
       
        self.prompt_pool=None
 
        self.text_model = AutoModel.from_config(config.text_config, attn_implementation=config._attn_implementation)

        self.image_seq_len = config.perceiver_config.resampler_n_latents
        self.image_token_id = self.config.image_token_id

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        final_ref_id_list = None,
        get_feature=0
    ) -> Union[Tuple, Idefics2BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        return_legacy_cache = False
        if use_cache:
            if not isinstance(past_key_values, Cache):  
                return_legacy_cache = True
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        
       
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids) 
       
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()

            
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

           
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values, 
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state
            
            
            image_hidden_states = self.connector(
                image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
            )

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)
        
       
        
        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
           
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids, 
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states, 
            )
        if position_ids is not None:
            pdb.set_trace()
        
        
        ref_idx=final_ref_id_list[0][1].item() 
        
        if self.prompt_pool is not None:
            res = self.prompt_pool(inputs_embeds[:,:ref_idx]) 
            inputs_embeds2=torch.concat([res['prompted_embedding'],inputs_embeds],dim=1)
        else:
            inputs_embeds2=inputs_embeds
            res=None
        
        
        new_attention_mask2= torch.ones((inputs_embeds2.shape[:2]),device=inputs_embeds.device)
        outputs = self.text_model(
            inputs_embeds=inputs_embeds2, 
           
            attention_mask=new_attention_mask2,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
       
            
        if return_legacy_cache and use_cache:
            outputs.past_key_values = outputs.past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return Idefics2BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        ), res

        

class Idefics2ForConditionalGenerationPreferTokens(Idefics2ForConditionalGeneration):
    def __init__(self, config,*model_args, **model_kwargs):
        super().__init__(config)
        
       
        self.model = Idefics2ModelPreferTokens(config, **model_kwargs)
        self.model_kwargs=model_kwargs
        self.image_token_id = self.config.image_token_id

        self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        
        self.post_init()
        
        
    @add_start_docstrings_to_model_forward(IDEFICS2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Idefics2CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        final_ref_id_list = None,
        get_feature=0,
    ) -> Union[Tuple, Idefics2CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics2ForConditionalGeneration`).
                Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
                computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:

        Example:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModelForVision2Seq
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
        >>> model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b-base", device_map="auto")

        >>> BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        >>> EOS_WORDS_IDS = [processor.tokenizer.eos_token_id]

        >>> # Create inputs
        >>> prompts = [
        ...   "<image>In this image, we can see the city of New York, and more specifically the Statue of Liberty.<image>In this image,",
        ...   "In which city is that bridge located?<image>",
        ... ]
        >>> images = [[image1, image2], [image3]]
        >>> inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda")

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=20)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts)
        ['In this image, we can see the city of New York, and more specifically the Statue of Liberty. In this image, we can see the city of New York, and more specifically the Statue of Liberty.\n\n', 'In which city is that bridge located?\n\nThe bridge is located in the city of Pittsburgh, Pennsylvania.\n\n\nThe bridge is']
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        outputs,res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            final_ref_id_list=final_ref_id_list,
            get_feature=False
        )
        
        
        
        hidden_states = outputs[0] 

        
        

        if get_feature:
            return  outputs.hidden_states, outputs.last_hidden_state
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()       
        self.total_prompt_len=res['total_prompt_len']
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            
           
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device)
                
                shift_logits = logits[..., self.total_prompt_len:-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., self.total_prompt_len:-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output



        return Idefics2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            reduce_sim=res['reduce_sim'] if 'reduce_sim' in res else None,
        )
        
        


local_rank = 0




DEVICE = "cuda:0" 
USE_LORA = False
USE_QLORA = False


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    use_dora=False if USE_QLORA else True,
    init_lora_weights="gaussian"
)   


     

def create_dataset(dataframe, ref_pair_num=2):
    dataset = defaultdict(dict)

    for index, row in dataframe.iterrows():
      
        dataset[index] = [(io.BytesIO(row['reference_list_bad'][i]), 
                           io.BytesIO(row['reference_list'][i]), 
                           row['reference_prompt_list'][i]) for i in random.sample(range(len(row['reference_list'])),ref_pair_num)] 
        if row['label0']>0.5:
            dataset[index].append((io.BytesIO(row['image1']), io.BytesIO(row['image0']),row['prompt'])) 
        elif row['label0']<0.5:
            dataset[index].append((io.BytesIO(row['image0']), io.BytesIO(row['image1']),row['prompt'])) 

    return dataset


class MyDataCollator:
    def __init__(self, processor,use_prompt=False,max_prompt_len=77):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.end_of_utterance_id = self.processor.tokenizer("<end_of_utterance>")["input_ids"][-1]
        self.use_prompt=use_prompt
        self.max_prompt_len=max_prompt_len
        self.flag=1


    def __call__(self, examples):
      
        texts, images, targets = [], [], []
        bz=len(examples)
        for example in examples: 
            curr_imgs, messages = [], []

            for negative_image, positive_image, ref_prompt in example[:-1]:          
                order = random.randint(0, 1) 
                imgs = [negative_image, positive_image]  if order % 2 == 0 else [positive_image, negative_image]
                score_list = ["-", "+"]  if order % 2 == 0 else ["+", "-"]
                for img, score in zip(imgs, score_list):
                    
                    curr_imgs.append(Image.open(img).resize((512,512)))
                    if self.use_prompt:
                        messages.append( 
                            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"The prompt is '{ref_prompt[:self.max_prompt_len]}' Score for this image?"}]}
                        )                    
                    else:
                        messages.append( 
                            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Score for this image?"}]}
                        )
                    messages.append(
                        {"role": "assistant", "content": [{"type": "text", "text": score}]}
                    )
           
            if self.use_prompt:
                messages.append( 
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"The prompt is '{example[-1][2][:self.max_prompt_len]}' Score for this image?"}]}
                )
            else:
                messages.append( 
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Score for this image?"}]}
                )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": ""}]}
            )
           
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)[:-19] 
            texts.extend([text.strip()] * 2) 
            images.append(curr_imgs.copy())
            images.append(curr_imgs.copy())
            
            targets.extend(["-", "+"]) 
            images[-2].append(Image.open(example[-1][0]).resize((512,512))) 
            images[-1].append(Image.open(example[-1][1]).resize((512,512)))
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True) 
        targets = self.processor(text=targets, return_tensors="pt", padding=True)

        batch["labels"] = batch["input_ids"].clone() 
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = self.image_token_id 
        

        
        ref_id_list_x=[]
        ref_id_list_y=[]
        mask = batch["input_ids"] == self.end_of_utterance_id
        indices = torch.where(mask) 
        for iii in range(bz*2):
            mask_iii = indices[0] == iii 
            indices_iii = torch.where(mask_iii)[0][-2] 
            ref_id_list_x.append(int(iii/bz))
            ref_id_list_y.append(indices[1][indices_iii])
        final_ref_id_list=torch.tensor([ref_id_list_x,ref_id_list_y]).T
        batch["final_ref_id_list"]=final_ref_id_list 
        
        batch={k:batch[k].contiguous().reshape(-1, 2, *batch[k].shape[1:]) for k in batch.keys()}

        batch["targets"] = targets["input_ids"][:, -1].reshape(bz,2,1)

 
        
        return batch


ds_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": False,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {

            "lr": 'auto',
            "weight_decay":"auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 0
    },


    "steps_per_print": 100,
    "train_batch_size": "auto",
    "gradient_accumulation_steps":"auto",
    "train_micro_batch_size_per_gpu": "auto",
    
    "wall_clock_breakdown": False
}



import json
with open("ds_config.json", "w") as f:
    json.dump(ds_config, f)

def train():
    global local_rank
    
    args = create_arg_parser()
    rank0_print(args)

    time_name=datetime.datetime.now().strftime("%m.%d_%H.%M.%S")

    run_name =  f"{args.exp_name}_"+ time_name 
    rank0_print(f"run_name: {run_name}")
    

    save_dir="ckpt"
    os.makedirs(save_dir,exist_ok=True)
    
    
    training_args = TrainingArguments(
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.train_bz,
        per_device_eval_batch_size=args.eval_bz,
        gradient_accumulation_steps=args.grad_acc_step,
        warmup_steps=0,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=1,
        output_dir=os.path.join(save_dir,run_name),
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        fp16=args.fp16, 
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="tensorboard",
        do_eval=True,
        deepspeed="ds_config.json",
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
    )

    if args.fp16:
        torch_dtype=torch.float16
    else:
        torch_dtype=torch.float32
   
    
    if getattr(training_args, "deepspeed", None) : 
        rank0_print("training_args: ",training_args)
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

   

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None

   
    if USE_QLORA:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype
        )
                

 
   
    processor = AutoProcessor.from_pretrained(
        idefics_path,
        do_image_splitting=False,
        size={"longest_edge": 448, "shortest_edge": 378},
        device_map=device_map,
    )
    
    if ddp:
        device=f'cuda:{int(os.environ.get("LOCAL_RANK") or 0)}'
    else:
        device="cuda"
    
    prompt_pool_args={"use_prompt_pool":0,
                      "pull_constraint_coeff":args.pull_constraint_coeff,
                      "device":device,
                      "pool_size":1, 
                      "prompt_top_k":1,"length":args.length,"batchwise_prompt":True,"embed_dim":4096}
    rank0_print(prompt_pool_args)
 
    model = Idefics2ForConditionalGenerationPreferTokens.from_pretrained(
        idefics_path,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config if USE_QLORA else None,
        device_map=device_map,
  
    )
    lora_path=args.lora_path
    model = PeftModel.from_pretrained(
        model, lora_path
    )
    model=model.base_model.model

    
    model.model.prompt_pool= PromptPool(**prompt_pool_args)

    model = model.to(device)

    
    rank0_print("prompt_pool.prompt.is_leaf:",model.model.prompt_pool.prompt.is_leaf)
 
    if USE_QLORA:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        
        
        
    for name, param in model.named_parameters():
        if "prompt_pool" in name:
            param.requires_grad=True
        else:
            param.requires_grad=False
            
    for name, param in model.named_parameters():
        if "prompt_pool" in name:
            rank0_print(f'{name}, requires_grad: {param.requires_grad}')
    
   
    rank0_print(get_parameter_number(model))

    rank0_print("Loading data...")
    with open("bench_val_w_bad1.pkl", 'rb') as f1:
        dataframe_val = pickle.load(f1)
    

    with open("bench_train_w_bad1.pkl", 'rb') as f1:
        dataframe_train = pickle.load(f1)


    
    dataframe_val = dataframe_train  
           
            
    train_set = create_dataset(dataframe_train, args.ref_pair_num)

    val_set = create_dataset(dataframe_val, args.ref_pair_num)
    
    rank0_print(f"train_set len: {len(train_set)}")

    rank0_print(f"val_set len: {len(val_set)}")    
    
    data_collator = MyDataCollator(processor,use_prompt=args.use_prompt,max_prompt_len=args.max_prompt_len)




    
    
    
    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset={"val":val_set},
    )
    
    trainer.train()
    trainer.save_state()


    

if __name__ == "__main__":
    train()





