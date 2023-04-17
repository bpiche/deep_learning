from transformers import GPT2TokenizerFast, GPT2LMHeadModel

print('Loading the GPT2 language model')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)


if __name__ == "__main__":
    """
    """
    corpus = '''Does Buy American apply to private projects, or private contractors on public projects?'''
    print(f'Generating text for the input:\n"{corpus}"\n')

    input_ids = tokenizer.encode(corpus, return_tensors="pt")

    tokenizer.convert_ids_to_tokens(input_ids[0])

    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(
                [corpus],
                padding=True
    )

    sample_output_top_k = model.generate(
        input_ids, 
        do_sample=True, 
        # min_length=100,
        max_length=50, 
        top_k=50, 
        # top_p=0.92, 
        temperature=0.08,
    )

    response = tokenizer.decode(sample_output_top_k[0], 
                                skip_special_tokens=True)

    print(f'Generated text:\n{response}')