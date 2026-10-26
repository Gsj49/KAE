"""Keep the conditioning prompt immutable when BPE merges across its boundary."""
import torch

POLICY = 'joined_prefix_if_stable_else_fixed_prompt_and_lossless_response_v1'


def tokenize_fixed_prompt(tokenizer, prompt_text, prompt_ids, responses):
    continuations, repairs = [], []
    prompt_ids = prompt_ids.detach().cpu().to(torch.long)
    prompt_length = int(prompt_ids.numel())
    for index, response in enumerate(responses):
        response = str(response)
        joined = tokenizer(prompt_text + response, return_tensors='pt')['input_ids'][0].detach().cpu()
        if joined.numel() >= prompt_length and torch.equal(joined[:prompt_length], prompt_ids):
            continuation = joined[prompt_length:].to(torch.long)
        else:
            # A generated continuation is conditioned on these exact prompt IDs.
            # Do not prepend the merged token containing part of the prompt again.
            continuation = tokenizer(response, add_special_tokens=False,
                                     return_tensors='pt')['input_ids'][0].detach().cpu().to(torch.long)
            decode_args = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if tokenizer.decode(prompt_ids, **decode_args) != prompt_text:
                raise ValueError('Prompt decoding is not lossless; refusing boundary repair')
            if tokenizer.decode(continuation, **decode_args) != response:
                raise ValueError('Response decoding is not lossless; refusing boundary repair')
            if tokenizer.decode(torch.cat([prompt_ids, continuation]), **decode_args) != prompt_text + response:
                raise ValueError('Reconstructed prompt/response differs from the saved text')
            shared = 0
            for left, right in zip(prompt_ids, joined):
                if int(left) != int(right):
                    break
                shared += 1
            repairs.append(dict(completion_index=index, shared_prefix_tokens=shared,
                prompt_tokens=prompt_length, joined_tokens=int(joined.numel()),
                continuation_tokens=int(continuation.numel()), policy=POLICY))
        if continuation.numel() == 0:
            raise ValueError(f'Empty continuation at completion {index}')
        continuations.append(continuation)
    return continuations, repairs
