
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
from peft import PeftModel
import pdb
import time
import os
import io
from torch.cuda.amp import autocast
import pickle
import pdb
from PIL import Image
from peft import PeftModel, PeftConfig
import io

from transformers import AutoProcessor, AutoModel
import os
from torch.cuda.amp import autocast
from prompt_pool import *

from train2 import Idefics2ForConditionalGenerationPreferTokens
import torch

import re
import time
import os
import random

def set_device(device='cuda:0'):
    torch.cuda.set_device(device)
    return device

device = DEVICE = set_device("cuda:0")

def load_context_images(negative_image_paths, positive_image_paths):
    context_images = []
    for i in range(len(negative_image_paths)):
        context_images.append(Image.open(negative_image_paths[i]).resize((512,512)))
        context_images.append(Image.open(positive_image_paths[i]).resize((512,512)))

    return context_images



def calculate_score(processor, model, context_images, query_image,ref_prompt="",tgt_prompt="",use_ref_id=False):
    prompt = ""

    if isinstance(ref_prompt, str):
        for i in range(len(context_images) // 2):
            prompt = prompt + "User:<image>Score for this image?<end_of_utterance>\n"
            prompt = prompt + "Assistant: -<end_of_utterance>\n"
            prompt = prompt + "User:<image>Score for this image?<end_of_utterance>\n"
            prompt = prompt + "Assistant: +<end_of_utterance>\n"
    elif isinstance(ref_prompt, list):
        for i in range(len(context_images) // 2):
            prompt = prompt + f"User:<image>The prompt is '{ref_prompt[i]}' Score for this image?<end_of_utterance>\n"
            prompt = prompt + "Assistant: -<end_of_utterance>\n"
            prompt = prompt + f"User:<image>The prompt is '{ref_prompt[i]}' Score for this image?<end_of_utterance>\n"
            prompt = prompt + "Assistant: +<end_of_utterance>\n"
            
                
    context_images.append(Image.open(query_image).resize((512,512)))

    if isinstance(ref_prompt, list) and tgt_prompt!="":
        prompt = prompt + f"User:<image>The prompt is '{tgt_prompt}' Score for this image?<end_of_utterance>\n"
    elif isinstance(ref_prompt, str):
        prompt = prompt + "User:<image>Score for this image?<end_of_utterance>\n"
    else:
        pdb.set_trace()
    prompt = prompt + "Assistant: "

    inputs = processor(text=prompt, images=context_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    
    
    if use_ref_id:
        ref_id_list_x=[]
        ref_id_list_y=[]
        mask = inputs["input_ids"] == processor.tokenizer("<end_of_utterance>")["input_ids"][-1]
        indices = torch.where(mask) 
        ref_id_list_x.append(0)
        ref_id_list_y.append(indices[1][-2])

        final_ref_id_list=torch.tensor([ref_id_list_x,ref_id_list_y]).T

        inputs["final_ref_id_list"]=final_ref_id_list 
              

    with torch.no_grad():
        with autocast( dtype=torch.float16):
            outputs = model(**inputs)

    logits = outputs.get("logits")

    # +: 648, -: 387
    
    score = torch.exp(logits[:, -1][:, 648]) / (torch.exp(logits[:, -1][:, 648]) + torch.exp(logits[:, -1][:, 387])).item()

    score = score.item()

    return score



lora_path="lora_path"

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    size={"longest_edge": 448, "shortest_edge": 378},
    do_image_splitting=False
)
torch_dtype =    torch.float16

    
model = Idefics2ForConditionalGenerationPreferTokens.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    ).to(device)

model = PeftModel.from_pretrained(
    model, lora_path
).to(device)



prompt_pool=torch.load("prompt_pool.pt")
model.base_model.model.model.prompt_pool=prompt_pool.to(device)




model.eval()



with open("bench_test_w_bad.pkl", 'rb') as f1:
    bench_test = pickle.load(f1)

tiic=time.time()
test_len=0
test_unseen_len=0
num=8
for index, row in bench_test.iterrows():

    tic=time.time()
    
    image0=[io.BytesIO(row['reference_list'][i]) for i in range(num)]

    ref_prompt=[row["reference_prompt_list"][i] for i in range(num)]
    tgt_prompt=row["prompt"]

    
    negative_image_paths=[io.BytesIO(row['reference_list_bad'][i]) for i in range(num)]
    label0=row["label0"]

 
    context_images = load_context_images(negative_image_paths, image0)

        
    context_images1=context_images.copy()
    query_image = io.BytesIO(row["image0"])
    score = calculate_score(processor, model, context_images1, query_image,ref_prompt,tgt_prompt,use_ref_id=True)
    # print(score)
