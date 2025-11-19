from typing import Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    def chat_model(self):
        raise NotImplementedError



class Qwen3(BaseLLM):
    DEFAULT_MODEL = "Qwen/Qwen3-8B"
    THINKING_EOS_TOKEN_ID = 151668  # token id for </think>

    def __init__(self, path: str = ""):
        super().__init__(path or self.DEFAULT_MODEL)
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    def chat_model(
        self,
        prompt: str,
        *,
        history: List[dict] | None = None,
        meta_instruction: str = "",
        max_new_tokens: int = 32768,
        enable_thinking: bool = True,
        **generate_kwargs,
    ) -> Tuple[Dict[str, str], List[dict]]:

        conversation = self.prepare_history(history, meta_instruction)
        conversation.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

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

        # CoT
        # thinking_content = self.tokenizer.decode(
        #     output_ids[:index],
        #     skip_special_tokens=True,
        # ).strip("\n")

        content = self.tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True,
        ).strip("\n")

        new_history = deepcopy(conversation)
        new_history.append({"role": "assistant", "content": content})
        return content, new_history