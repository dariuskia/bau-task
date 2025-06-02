import os
from IPython.display import clear_output

from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG
import torch

load_dotenv()
CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))

clean_prompt = """Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [dog apple cherry bus cat grape bowl]
Answer: ("""
corrupted_prompt = """Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [dog apple spoon bus cat grape bowl]
Answer: ("""
prompts = (clean_prompt, corrupted_prompt)

answers = ("3", "2")

model = LanguageModel("meta-llama/Meta-Llama-3.1-70B")

tokens = model.tokenizer(prompts, return_tensors="pt")['input_ids']
print(f"{tokens=}")
correct_idx, incorrect_idx = [model.tokenizer(answer, add_special_tokens=False)['input_ids'][0] for answer in answers]

N_LAYERS = len(model.model.layers)
with model.trace(remote=True) as tracer:
    with tracer.invoke(clean_prompt) as invoker:
        clean_hs = [model.model.layers[layer_idx].output[0] for layer_idx in range(N_LAYERS)]
        clean_logits = model.lm_head.output

         # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (clean_logits[0, -1, correct_idx] - clean_logits[0, -1, incorrect_idx]).save()

    with tracer.invoke(corrupted_prompt) as invoker:
        corrupted_logits = model.lm_head.output

         # Calculate the difference between the correct answer and incorrect answer for the corrupted run.
        corrupted_logit_diff = (corrupted_logits[0, -1, correct_idx] - corrupted_logits[0, -1, incorrect_idx]).save()

    ioi_patching_results = []
    for layer_idx in range(N_LAYERS):
        _ioi_patching_results = []

        # Iterate through all tokens
        for token_idx in range(len(tokens[-1])):
            # Patching corrupted run at given layer and token
            with tracer.invoke(corrupted_prompt) as invoker:
                # Apply the patch from the clean hidden states to the corrupted hidden states.
                model.model.layers[layer_idx].output[0][:, token_idx, :] = clean_hs[layer_idx][:, token_idx, :]

                patched_logits = model.lm_head.output

                patched_logit_diff = (
                    patched_logits[0, -1, correct_idx]
                    - patched_logits[0, -1, incorrect_idx]
                )

                # Calculate the improvement in the correct token after patching.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )

                _ioi_patching_results.append(patched_result.item().save())

        ioi_patching_results.append(_ioi_patching_results)

torch.save(ioi_patching_results, "/workspace/bau/ioi_patching_results.pt")
