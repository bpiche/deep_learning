from rasterize_paper import rasterize_paper
import argparse
import torch
import re

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="data/lilt_lm.pdf")
parser.add_argument("--prompt", type=str, default="What is the title of this paper?")
args = parser.parse_args()

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


if __name__ == "__main__":
    """
    """
    filepath = args.filename
    images = rasterize_paper(filepath, return_pil=True)
    image = Image.open(images[0])
    pixel_values = processor(image, return_tensors="pt").pixel_values

    task_prompt = args.prompt
    question = "When is the coffee break?"
    prompt = task_prompt.replace("{user_input}", question)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    
    outputs = model.generate(pixel_values.to(device),
                                decoder_input_ids=decoder_input_ids.to(device),
                                max_length=model.decoder.config.max_position_embeddings,
                                early_stopping=True,
                                pad_token_id=processor.tokenizer.pad_token_id,
                                eos_token_id=processor.tokenizer.eos_token_id,
                                use_cache=True,
                                num_beams=1,
                                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                return_dict_in_generate=True,
                                output_scores=True)
    
    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # remove the prompt query from the beginning of the sequence
    seq = re.sub(prompt, "", seq, count=1).strip()
    # remove the first question mark
    seq = re.sub(r"^\?", "", seq, count=1).strip()
    # remove everything after the second question mark
    seq = re.sub(r"\?.*$", "", seq, count=1).strip()
    print(f'prompt: {prompt}\nresponse: {seq}')
        