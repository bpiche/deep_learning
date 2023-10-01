from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch

from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict

from rasterize_paper import rasterize_paper
from PIL import Image


processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# custom stopping criteria from the nougat authors
class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


# prepare PDF image for the model
filepath = "data/lilt_lm.pdf"
images = rasterize_paper(pdf=filepath, return_pil=True)

output_md = filepath.replace(".pdf", ".md")
for i in images:
    image = Image.open(i)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # autoregressively generate tokens, with custom stopping criteria (as defined by the Nougat authors)
    outputs = model.generate(pixel_values.to(device),
                            min_length=1,
                            max_length=3584,
                            bad_words_ids=[[processor.tokenizer.unk_token_id]],
                            return_dict_in_generate=True,
                            output_scores=True,
                            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
    )
    sequence = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
    sequence = processor.post_process_generation(sequence, fix_markdown=False)
    # write the generated markdown to a file
    with open(output_md, "a") as f:
        f.write(sequence)