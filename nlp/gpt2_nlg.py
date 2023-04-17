import os
import pandas as pd
from transformers import (
    AutoModelWithLMHead,
    AutoConfig,
    Trainer,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments)

from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer

# from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# we're doing this again later in finetune_model(), no need to do it twice
# print('Loading the GPT2 language model')
# tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)


def finetune_model(text_path, epochs, model='gpt2', batch_size=8, cache_dir='cache'):
    """
    """
    model = AutoModelWithLMHead.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_path,
        block_size=256,)
    training_args = TrainingArguments(
        output_dir="gpt2_fine_fune/{}".format(os.path.basename(text_path)),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        save_steps=2000,
        logging_steps=10)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # num_samples=1
        # prediction_loss_only=True,
        )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    """
    """
    # corpus = '''Does Buy American apply to private projects, or private contractors on public projects?'''
    # print(f'Generating text for the input:\n"{corpus}"\n')
    # input_ids = tokenizer.encode(corpus, return_tensors="pt")
    # tokenizer.convert_ids_to_tokens(input_ids[0])
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenized = tokenizer(
    #             [corpus],
    #             padding=True)
    # sample_output_top_k = model.generate(
    #     input_ids, 
    #     do_sample=True, 
    #     # min_length=100,
    #     max_length=50, 
    #     top_k=50, 
    #     # top_p=0.92, 
    #     temperature=0.08)
    # response = tokenizer.decode(sample_output_top_k[0], 
    #                             skip_special_tokens=True)
    # print(f'Generated text:\n{response}')
    df = pd.read_csv('finetune.csv')[['original','modern']]
    df['combined'] = '<s>' + df.modern + '</s>' + '>>>>' + '<p>' + df.original + '</p>'
    df['combined'] = df.combined.to_csv('combined.txt', sep='\n', index=False)
    finetune_model('combined.txt', epochs=1, batch_size=8)