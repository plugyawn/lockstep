# Available Checkpoints for DUO and D2F Models

## DUO Checkpoints (from s-sahoo)

### HuggingFace Models
1. **Distilled Model**: `s-sahoo/duo-distilled`
   - Optimized for few-step generation (8 steps)
   - Trained on OpenWebText for 1M steps
   - ~130M parameters (GPT2-medium size)
   - Context length: 1024 tokens

2. **Un-distilled Model**: `s-sahoo/duo`
   - Full model without distillation
   - Same architecture as distilled version
   - Better quality but slower generation

### Usage
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load distilled model (faster, 8-step generation)
model = AutoModelForMaskedLM.from_pretrained('s-sahoo/duo-distilled')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Load un-distilled model (slower, better quality)
model = AutoModelForMaskedLM.from_pretrained('s-sahoo/duo')
```

### Additional Resources
- **Google Drive**: Contains `.ckpt` format checkpoints for fine-tuning
- **Collection**: https://huggingface.co/collections/s-sahoo/duo-67f9ff8fde919224e5fbd875

---

## D2F Checkpoints (from SJTU-Deng-Lab / zhijie-group)

### HuggingFace LoRA Models

1. **D2F-Dream-Base-7B-Lora**
   - URL: `SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora`
   - Based on Dream-7B architecture
   - LoRA adaptation for efficient fine-tuning
   - Enables 2.5× speedup over LLaMA3

2. **D2F-LLaDA-Instruct-8B-Lora**
   - URL: `SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora`
   - Instruction-tuned variant
   - 8B parameter model
   - 50× acceleration vs vanilla dLLMs

### Usage
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base Dream-7B model
base_model = AutoModelForCausalLM.from_pretrained("kuleshov-group/dream-7b")

# Apply D2F LoRA weights
d2f_model = PeftModel.from_pretrained(
    base_model,
    "SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora"
)
```

### Additional Resources
- **GitHub**: https://github.com/zhijie-group/Discrete-Diffusion-Forcing
- **Demo**: https://huggingface.co/spaces/zhijie3/D2F-LLaDA-Instruct-8B
- **Blog**: https://zhijie-group.github.io/Discrete-Diffusion-Forcing/

---

## Comparison for Lockstep R/A

| Model | Training Required | Speedup | Architecture Change | Best For |
|-------|------------------|---------|-------------------|----------|
| DUO-distilled | None (pre-trained) | 100× | None | Quick integration |
| D2F-Dream-7B-LoRA | None (LoRA weights) | 50× | Block-causal | Production with KV cache |

## Recommendations

1. **For immediate use**: Use `s-sahoo/duo-distilled` - it's ready to use with minimal setup
2. **For Dream-7B compatibility**: Use `SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora` - directly compatible with Dream-7B
3. **For best speed**: DUO-distilled (100× speedup with 8-step generation)
4. **For production**: D2F with LoRA (KV cache support, block-causal attention)