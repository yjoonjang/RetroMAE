"""Benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""
from __future__ import annotations

import os
import logging
from multiprocessing import Process, current_process
import torch
import hashlib
import torch._dynamo
torch._dynamo.config.disable = True

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import mteb
from mteb import MTEB, get_tasks
from mteb.encoder_interface import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.models.instruct_wrapper import instruct_wrapper

import argparse
from dotenv import load_dotenv
from setproctitle import setproctitle
import traceback
import logging

load_dotenv() # for OPENAI

# AICA 한정
# os.environ['HF_HOME'] = '/data/EMBEDDING/cache'
# os.environ['TRANSFORMERS_CACHE'] = '/data/EMBEDDING/cache'

parser = argparse.ArgumentParser(description="Extract contexts")
parser.add_argument('--quantize', default=False, type=bool, help='quantize embeddings')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

# MIRACL, MrTidy는 평가 시 시간이 오래 걸리기 때문에, 태스크별로 나누어 multiprocessing으로 평가합니다.
# 필요 시 GPU 번호를 다르게 조정해 주세요.

# TASK_LIST_RETRIEVAL_GPU_MAPPING = {
#     4: [
#         "Ko-StrategyQA",
#         "AutoRAGRetrieval",
#         "PublicHealthQA",
#         "BelebeleRetrieval",
#         "XPQARetrieval",
#         # "KoFinMarketReportRetrieval",
#         # "KoFSSFinDictRetrieval",
#         # "KoSquadv1Retrieval",
#         # "KoTATQARetrieval",
#         "MultiLongDocRetrieval",
#     ],
#     5: ["MIRACLRetrieval"],
#     0: ["MrTidyRetrieval"],
# }
TASK_LIST_RETRIEVAL_GPU_MAPPING = {
    6: [
        "Ko-StrategyQA",
        "AutoRAGRetrieval",
        "PublicHealthQA",
        "BelebeleRetrieval",
        "XPQARetrieval",
        # "KoFinMarketReportRetrieval",
        # "KoFSSFinDictRetrieval",
        # "KoSquadv1Retrieval",
        # "KoTATQARetrieval",
        "MultiLongDocRetrieval",
        "MrTidyRetrieval",
    ],
    7: ["MIRACLRetrieval"],
}

model_names = [
    # my_model_directory
]
model_names = [
    # "/mnt/raid6/yjoonjang/projects/RetroMAE/MODELS/checkpoint-375"
    # "skt/A.X-Encoder-base"
    # "/mnt/raid6/yjoonjang/projects/RetroMAE/MODELS/skt_A.X-Encoder-base-dupmae"
    # "BAAI/bge-m3-retromae"
    # "/mnt/raid6/yjoonjang/projects/RetroMAE/MODELS/skt_A.X-Encoder-base-dupmae-8192"
    "/mnt/raid6/yjoonjang/projects/RetroMAE/MODELS/skt_A.X-Encoder-base-retromae-8192"
] + model_names

save_path = "./RESULTS"

def evaluate_model(model_name, gpu_id, tasks):
    import torch
    try:
        # CUDA 디바이스 설정을 더 명확하게
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # torch.cuda.empty_cache()
        # device = torch.device(f"cuda:0")
        # torch.cuda.set_device(device)
        device = torch.device(f"cuda:{str(gpu_id)}") 
        torch.cuda.set_device(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        
        model = None
        if not os.path.exists(model_name): # hf에 등록된 모델의 경우
            if "m2v" in model_name: # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                static_embedding = StaticEmbedding.from_model2vec(model_name)
                model = SentenceTransformer(modules=[static_embedding], model_kwargs={"attn_implementation": "sdpa"}, device=device)
            else:
                if model_name == "nlpai-lab/KoE5" or model_name == "KU-HIAI-ONTHEIT/ontheit-large-v1_1" or "KUKE" in model_name:
                    # mE5 기반의 모델이므로, 해당 프롬프트를 추가시킵니다.
                    model_prompts = {
                        PromptType.query.value: "query: ",
                        PromptType.passage.value: "passage: ",
                    }
                    model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"}, device=device)
                elif "snowflake" in model_name.lower():
                    model_prompts = {
                        PromptType.query.value: "query: ",
                    }
                    model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, device=device)
                elif "Qwen3" in model_name:
                    # model_prompts = {
                    #     # PromptType.query.value: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
                    #     PromptType.query.value: "Instruct: 주어진 질의로 해당 질의를 답변할 수 있는 문서를 검색하세요.\n질의:",
                    # }
                    # model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}, device=device, tokenizer_kwargs={"padding_side": "left"})
                    model = mteb.get_model(model_name, model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}, device=device, tokenizer_kwargs={"padding_side": "left"},)
                elif "frony" in model_name:
                    model_prompts = {
                        # PromptType.query.value: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
                        PromptType.query.value: "<Q>",
                        PromptType.passage.value: "<P>",
                    }
                    model = mteb.get_model(model_name, model_kwargs={"attn_implementation": "sdpa"}, device=device)
                elif "gte-multilingual" in model_name or "nomic-embed" in model_name:
                    model = mteb.get_model(model_name, trust_remote_code=True, device=device)
                else:
                    # mteb에 등록된 모델의 경우, 프롬프트/prefix 등을 포함하여 평가할 수 있습니다. 등록되지 않은 경우, sentence-transformer를 사용하여 불러옵니다.
                    # model = mteb.get_model(model_name, device=device)
                    model = SentenceTransformerWrapper(
                        model=model_name, 
                        model_kwargs={"torch_dtype": torch.bfloat16}, 
                        device=device, 
                        tokenizer_kwargs={"model_max_length": 8192}
                    )

                    if hasattr(model.model.tokenizer, 'model_input_names') and 'token_type_ids' in model.model.tokenizer.model_input_names:
                        model.model.tokenizer.model_input_names = [name for name in model.model.tokenizer.model_input_names if name != 'token_type_ids']
                    model.model._modules["1"].pooling_mode_mean_tokens = False
                    model.model._modules["1"].pooling_mode_cls_token = True
                    logging.info("using CLS pooling ...")
        else: # 직접 학습한 모델의 경우
            file_name = os.path.join(model_name, "model.safetensors")
            if os.path.exists(file_name):
                if "m2v" in model_name: # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                    static_embedding = StaticEmbedding.from_model2vec(model_name)
                    model = SentenceTransformer(modules=[static_embedding], model_kwargs={"attn_implementation": "sdpa"}, device=device)
                # elif "v2_ko_only" in model_name:
                #     model_prompts = {
                #         PromptType.query.value: "사용자 질문을 증거 문서 검색을 위해 표현하세요: ",
                #     }
                #     model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"}, device=device)
                elif "v2_ko_en" in model_name:
                    model_prompts = {
                        PromptType.query.value: "Represent the query for retrieving evidence documents: ",
                    }
                    model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, device=device)
                else:
                    # model = mteb.get_model(model_name, model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}, device=device)
                    # model = SentenceTransformerWrapper(model=model_name, model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}, device=device)
                    model = SentenceTransformerWrapper(
                        model=model_name, 
                        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16}, 
                        device=device, 
                        tokenizer_kwargs={"model_max_length": 8192}
                    )

                    if hasattr(model.model.tokenizer, 'model_input_names') and 'token_type_ids' in model.model.tokenizer.model_input_names:
                        model.model.tokenizer.model_input_names = [name for name in model.model.tokenizer.model_input_names if name != 'token_type_ids']
                    model.model._modules["1"].pooling_mode_mean_tokens = False
                    model.model._modules["1"].pooling_mode_cls_token = True

        if model:
            # Create a shorter name for long model paths to avoid OS errors
            output_folder_name = os.path.basename(model_name)
            if os.path.isdir(model_name) and len(output_folder_name) > 100:
                model_hash = hashlib.md5(model_name.encode()).hexdigest()[:6]
                output_folder_name = f"{output_folder_name[:93]}_{model_hash}"

            # For local models, mteb can create subdirectories with excessively long names.
            # Overriding model_meta.name with our shorter name prevents this.
            if os.path.isdir(model_name):
                try:
                    # This assumes the model object from mteb has this attribute.
                    model.model_meta.name = output_folder_name
                except AttributeError:
                    logger.warning("Could not override model_meta.name. Path might still be too long.")
            
            setproctitle(f"{output_folder_name}-{gpu_id}")
            print(f"Running tasks: {tasks} / {model_name} on GPU {gpu_id} in process {current_process().name}")
            evaluation = MTEB(
                tasks=get_tasks(tasks=tasks, languages=["kor-Kore", "kor-Hang", "kor_Hang"])
            )
            # 48GB VRAM 기준 적합한 batch sizes
            if "multilingual-e5" in model_name or "KoE5" in model_name or "ontheit" in model_name or "KUKE" in model_name:
                batch_size = 512
            elif "jina" in model_name:
                batch_size = 8
            elif "bge-m3" in model_name or "Snowflake" in model_name:
                batch_size = 64
            elif "gemma2" in model_name:
                batch_size = 256 
            elif "Salesforce" in model_name:
                batch_size = 8
            else:
                batch_size = 16

            if args.quantize: # quantized model의 경우
                evaluation.run(
                    model,
                    output_folder=f"{save_path}/{output_folder_name}-quantized",
                    encode_kwargs={"batch_size": batch_size, "precision": "binary"},
                )
            else:
                evaluation.run(
                    model,
                    output_folder=f"{save_path}/{output_folder_name}",
                    encode_kwargs={"batch_size": batch_size},
                )
    except Exception as ex:
        print(ex)
        traceback.print_exc()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    for model_name in model_names:
        print(f"Starting evaluation for model: {model_name}")
        processes = []
        
        for gpu_id, tasks in TASK_LIST_RETRIEVAL_GPU_MAPPING.items():
            p = Process(target=evaluate_model, args=(model_name, gpu_id, tasks))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        print(f"Completed evaluation for model: {model_name}")