from app.services.llm.base_llm import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class TransformersLLM(BaseLLM):
    def __init__(self, model_id: str, device: str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        # For small models only. For bigger models or Llama you may need accelerate / device_map.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16 if self.device=="cuda" else None, device_map="auto" if self.device=="cuda" else None)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device=="cuda" else -1)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        out = self.generator(prompt, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"]
