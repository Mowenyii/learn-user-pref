import logging
import os
import torch
from accelerate.utils import DistributedType
from transformers.integrations import deepspeed

from peft import LoraConfig, prepare_model_for_kbit_training
import random

from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
import pickle
import io
import datetime
import argparse
import numpy as np







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
        type=bool,
        default=False,
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
    args=parser.parse_args()
    return args


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        def process_inputs(inputs, idx):
            sub_inputs = {key: val[:,idx:idx+1,:].contiguous().reshape(val.shape[0], *val.shape[2:]) for key, val in inputs.items()}
            targets = sub_inputs.pop("targets").squeeze(1)
       
            logits = model(**sub_inputs).get("logits") 
            
            return logits, targets 
        
        negative_logits, negative_targets = process_inputs(inputs, 0)
        positive_logits, positive_targets = process_inputs(inputs, 1)

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




        torch.cuda.empty_cache()
        return (loss, negative_logits) if return_outputs else loss



local_rank = 0




DEVICE = "cuda:0" 
USE_LORA = False
USE_QLORA = True


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    use_dora=False if USE_QLORA else True,
    init_lora_weights="gaussian"
)   


    
     
# Create train and test sets
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

# Data Collator class
class MyDataCollator:
    def __init__(self, processor,use_prompt=False,max_prompt_len=77):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.use_prompt=use_prompt
        self.max_prompt_len=max_prompt_len


    def __call__(self, examples):
        texts, images, targets = [], [], []
        bz=len(examples)

        for example in examples: # batch 
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
        batch={k:batch[k].contiguous().reshape(-1, 2, *batch[k].shape[1:]) for k in batch.keys()} 
        batch["targets"] = targets["input_ids"][:, -1].reshape(bz,2,1)

        return batch



def train():
    global local_rank
    
    args = create_arg_parser()
    time_name=datetime.datetime.now().strftime("%m.%d_%H.%M.%S")
    
    run_name =  f"{args.exp_name}_"+ time_name 
    rank0_print(f"run_name: {run_name}")
    rank0_print(f"root_path: {root_path}")
    

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
        fp16=True,  
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="tensorboard",
        do_eval=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
    )


    if getattr(training_args, "deepspeed", None) : 
        rank0_print("??args: ",training_args)
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
            bnb_4bit_compute_dtype=torch.float16
        )
                


    processor = AutoProcessor.from_pretrained(
        idefics_path,
        do_image_splitting=False,
        size={"longest_edge": 448, "shortest_edge": 378},
        device_map=device_map,
    )

    model = Idefics2ForConditionalGeneration.from_pretrained(
        idefics_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config if USE_QLORA else None,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    if USE_QLORA:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    model.add_adapter(lora_config)
    model.enable_adapters()
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        
    for name, param in model.named_parameters():
        rank0_print(f'{name, param.requires_grad}')
    
    
    rank0_print(get_parameter_number(model))

    rank0_print("Loading data...")
    with open("bench_val_w_bad.pkl", 'rb') as f1:
        dataframe_val = pickle.load(f1)
        
    with open("bench_train_w_bad.pkl", 'rb') as f1:
        dataframe_train = pickle.load(f1)
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

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,)
        
   
    
    
    

if __name__ == "__main__":
    train()





