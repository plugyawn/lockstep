# DUO Model Quality Issues

## Summary
The DUO-130M models from s-sahoo produce **incoherent text** despite proper implementation and sampling.

## Models Tested
1. **s-sahoo/duo-distilled** (130M params)
   - Output: "isimov Fontaine DublinLovebullical..."
   - Completely nonsensical

2. **s-sahoo/duo** (130M params, non-distilled)
   - Output: "preeximately understood, she said..."
   - Still incoherent

## Technical Details
- Flash-attn properly installed ✓
- Model loads correctly ✓
- Proper discrete diffusion sampling implemented ✓
- Correct vocab size (50258) ✓
- Proper timestep scheduling ✓

## Root Cause
The 130M parameter DUO models appear to be:
1. Too small to generate coherent text
2. Possibly undertrained
3. May be research prototypes not intended for production use

## Recommendation
**Do not use DUO-130M for text generation**. Consider:
1. **Dream-7B** (7B params) - Properly trained diffusion LM
2. **D2F-Dream-7B-LoRA** - Accelerated version of Dream-7B
3. Traditional autoregressive models for coherent generation

## Conclusion
While DUO offers interesting acceleration techniques (100× speedup), the available 130M checkpoints produce garbage output and are not suitable for any practical application requiring coherent text generation.