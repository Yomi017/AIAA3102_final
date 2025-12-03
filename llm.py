from typing import Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
from loguru import logger

# 禁用 tokenizers 并行化以避免 vLLM 多进程中的死锁警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

class BaseLLM:
    def __init__(self, path: str = ""):
        self.path = path
        self.supports_multimodal = False  # 默认不支持多模态

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

    def __init__(self, path: str = DEFAULT_MODEL, gpu_ids: List[int] = None):
        # 如果指定了GPU序号,设置CUDA_VISIBLE_DEVICES并调整tensor_parallel_size
        if gpu_ids is not None:
            gpu_str = ",".join(map(str, gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            self.tensor_parallel_size = len(gpu_ids)
            print(f"Using GPUs: {gpu_str} (mapped to {self.tensor_parallel_size} devices)")
        else:
            # gpu_ids 为 None 时自动检测并使用全部 GPU
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                self.tensor_parallel_size = gpu_count if gpu_count > 0 else 1
                print(f"Using all available GPUs: {self.tensor_parallel_size} device(s)")
            except:
                self.tensor_parallel_size = 1
                print("Could not detect GPU count, using tensor_parallel_size=1")
        
        self.use_vllm = False
        super().__init__(path or self.DEFAULT_MODEL)
        self.supports_multimodal = False  # 纯文本模型
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
        images: List[str] | None = None,
        max_new_tokens: int = 32768,
        enable_thinking: bool = True,
        **generate_kwargs,
    ) -> Tuple[Dict[str, str], List[dict]]:
        if images:
            logger.warning(f"Qwen3 does not support multimodal input. Ignoring {len(images)} image(s).")
        
        logger.debug(f"LLM chat called with prompt length: {len(prompt)}")
        start_time = time.time()

        conversation = self.prepare_history(history, meta_instruction)
        if prompt != "":
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


class Qwen3VL(BaseLLM):
    DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
    THINKING_EOS_TOKEN_ID = 151668  # token id for </think>
    
    def __init__(self, path: str = "", gpu_ids: List[int] = None, gpu_memory_utilization: float = 0.9):
        # 如果指定了GPU序号,设置CUDA_VISIBLE_DEVICES并调整tensor_parallel_size
        if gpu_ids is not None:
            gpu_str = ",".join(map(str, gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            self.tensor_parallel_size = len(gpu_ids)
            logger.info(f"Using GPUs: {gpu_str} (mapped to {self.tensor_parallel_size} devices)")
        else:
            # gpu_ids 为 None 时自动检测并使用全部 GPU
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                self.tensor_parallel_size = gpu_count if gpu_count > 0 else 1
                logger.info(f"Using all available GPUs: {self.tensor_parallel_size} device(s)")
            except:
                self.tensor_parallel_size = 1
                logger.warning("Could not detect GPU count, using tensor_parallel_size=1")
        
        self.use_vllm = False
        super().__init__(path or self.DEFAULT_MODEL)
        self.supports_multimodal = True  # 支持多模态
        self.gpu_memory_utilization = gpu_memory_utilization
        self.load_model()
    
    def load_model(self):
        logger.info(f"Loading VL model from: {self.path}")
        start_time = time.time()
        
        # 尝试使用 vLLM
        try:
            from vllm import LLM, SamplingParams
            self.use_vllm = True
            self.vllm_class = LLM
            self.sampling_params_class = SamplingParams
            logger.info(f"vLLM detected. Initializing VL model with tensor_parallel_size={self.tensor_parallel_size}")
        except ImportError:
            logger.info("vLLM not found. Falling back to transformers for VL model.")
            self.use_vllm = False
        
        # 加载 tokenizer/processor
        try:
            from transformers import AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            self.process_vision_info = process_vision_info
            
            logger.debug("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.path,
                trust_remote_code=True
            )
            logger.success("Processor loaded")
            
        except ImportError as e:
            logger.error(f"Failed to import required VL libraries: {e}")
            raise ImportError(
                "Please install required packages: pip install qwen-vl-utils transformers"
            )
        
        # 根据后端加载模型
        if self.use_vllm:
            logger.info("Initializing vLLM VL model...")
            self.model = self.vllm_class(
                model=self.path,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=32768,  # 增加最大长度以支持更长的对话
                limit_mm_per_prompt={"image": 10, "video": 10},  # 多模态限制
            )
        else:
            logger.info("Initializing transformers VL model...")
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            ).eval()
        
        elapsed = time.time() - start_time
        logger.success(f"VL model loaded successfully in {elapsed:.2f}s (backend: {'vLLM' if self.use_vllm else 'transformers'})")
    
    def chat(
        self,
        prompt: str,
        *,
        history: List[dict] | None = None,
        meta_instruction: str = "",
        images: List[str] | None = None,
        max_new_tokens: int = 32768,
        enable_thinking: bool = True,
        **generate_kwargs,
    ) -> Tuple[str, List[dict]]:
        """
        多模态对话
        Args:
            prompt: 文本提示
            history: 对话历史
            meta_instruction: 系统指令
            images: 图片路径列表
            max_new_tokens: 最大生成token数
            enable_thinking: 是否启用thinking模式
        Returns:
            (生成的文本, 更新后的历史)
        """
        logger.debug(f"VL chat called with prompt length: {len(prompt)}, images: {len(images) if images else 0}")
        start_time = time.time()
        
        conversation = self.prepare_history(history, meta_instruction)
        
        # 构建多模态消息
        content = []
        if images:
            for img_path in images:
                content.append({"type": "image", "image": img_path})
        if prompt != "":
            content.append({"type": "text", "text": prompt})
        
        if len(content) > 0:
            conversation.append({"role": "user", "content": content})

        if self.use_vllm:
            # vLLM 路径
            # 处理输入格式
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            # vLLM 多模态输入格式
            mm_data = {}
            if images:
                mm_data["image"] = images
            
            sampling_params = self.sampling_params_class(
                max_tokens=max_new_tokens,
                temperature=generate_kwargs.get("temperature", 0.7),
                top_p=generate_kwargs.get("top_p", 0.8),
                top_k=generate_kwargs.get("top_k", 20),
                stop_token_ids=[self.processor.tokenizer.eos_token_id] + generate_kwargs.get("stop_token_ids", []),
            )
            
            outputs = self.model.generate(
                {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                },
                sampling_params=sampling_params,
            )
            
            output_text = outputs[0].outputs[0].text.strip()
            
        else:
            # transformers 路径
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            image_inputs, video_inputs = self.process_vision_info(conversation)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )
            
            # 处理thinking token
            output_ids = generated_ids[0][inputs.input_ids.shape[-1]:].tolist()
            
            try:
                index = len(output_ids) - output_ids[::-1].index(self.THINKING_EOS_TOKEN_ID)
            except ValueError:
                index = 0
            
            output_text = self.processor.decode(
                output_ids[index:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        
        # 更新历史 - 只保存文本内容
        new_history = deepcopy(conversation)
        new_history.append({"role": "assistant", "content": output_text})
        
        elapsed = time.time() - start_time
        logger.info(f"VL model generated response in {elapsed:.2f}s (length: {len(output_text)} chars)")
        
        return output_text, new_history