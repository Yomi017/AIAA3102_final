from typing import Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
from loguru import logger

class BaseLLM:
    def __init__(self, path: str = ""):
        self.path = path

    def load_model(self):
        raise NotImplementedError
    
    def prepare_history(self, history: List[dict], meta_instruction: str) -> List[dict]:
        new_history = deepcopy(history) if history else []
        if meta_instruction and not any(m.get("role") == "system" for m in new_history):
            new_history.insert(0, {"role": "system", "content": meta_instruction})
        return new_history
    
    def chat(self):
        raise NotImplementedError



class Qwen3(BaseLLM):
    DEFAULT_MODEL = "Qwen/Qwen3-8B"
    THINKING_EOS_TOKEN_ID = 151668  # token id for </think>

    def __init__(self, path: str = "", tensor_parallel_size: int = 8, gpu_ids: List[int] = None):
        # 如果指定了GPU序号,设置CUDA_VISIBLE_DEVICES并调整tensor_parallel_size
        if gpu_ids is not None:
            gpu_str = ",".join(map(str, gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            self.tensor_parallel_size = len(gpu_ids)
            print(f"Using GPUs: {gpu_str} (mapped to {self.tensor_parallel_size} devices)")
        else:
            self.tensor_parallel_size = tensor_parallel_size
        
        self.use_vllm = False
        super().__init__(path or self.DEFAULT_MODEL)
        self.load_model()

    def load_model(self):
        logger.info(f"Loading model from: {self.path}")
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams
            self.use_vllm = True
            self.vllm_class = LLM
            self.sampling_params_class = SamplingParams
            logger.info(f"vLLM detected. Initializing with tensor_parallel_size={self.tensor_parallel_size}")
            print(f"vLLM detected. Initializing with tensor_parallel_size={self.tensor_parallel_size}")
        except ImportError:
            logger.info("vLLM not found. Falling back to transformers.")
            print("vLLM not found. Falling back to transformers.")
            self.use_vllm = False

        logger.debug("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        logger.success("Tokenizer loaded")

        # self.use_vllm = False

        if self.use_vllm:
            logger.info("Initializing vLLM model...")
            self.model = self.vllm_class(
                model=self.path,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=0.5,
            )
        else:
            logger.info("Initializing transformers model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
        
        elapsed = time.time() - start_time
        logger.success(f"Model loaded successfully in {elapsed:.2f}s (backend: {'vLLM' if self.use_vllm else 'transformers'})")

    def chat(
        self,
        prompt: str,
        *,
        history: List[dict] | None = None,
        meta_instruction: str = "",
        max_new_tokens: int = 32768,
        enable_thinking: bool = True,
        **generate_kwargs,
    ) -> Tuple[Dict[str, str], List[dict]]:
        logger.debug(f"LLM chat called with prompt length: {len(prompt)}")
        start_time = time.time()

        conversation = self.prepare_history(history, meta_instruction)
        conversation.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        if self.use_vllm:
            sampling_params = self.sampling_params_class(
                max_tokens=max_new_tokens,
                temperature=generate_kwargs.get("temperature", 0.7),
                top_p=generate_kwargs.get("top_p", 0.8),
                top_k=generate_kwargs.get("top_k", 20),
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id] + generate_kwargs.get("stop_token_ids", []),
            )
            outputs = self.model.generate([text], sampling_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text
            
            content = generated_text.strip()
            
        else:
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            prompt_length = model_inputs["input_ids"].shape[-1]

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

            output_ids = generated_ids[0][prompt_length:].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(self.THINKING_EOS_TOKEN_ID)
            except ValueError:
                index = 0

            content = self.tokenizer.decode(
                output_ids[index:],
                skip_special_tokens=True,
            ).strip("\n")

        new_history = deepcopy(conversation)
        new_history.append({"role": "assistant", "content": content})
        
        elapsed = time.time() - start_time
        logger.info(f"LLM generated response in {elapsed:.2f}s (length: {len(content)} chars)")
        
        return content, new_history