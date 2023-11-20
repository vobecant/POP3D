import clip
import torch

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


class PromptExtractor(torch.nn.Module):
    def __init__(self, templates=VILD_PROMPT):
        super().__init__()
        self.templates = templates

    def forward(self, noun_list, clip_model: torch.nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            text_inputs = torch.cat(noun_tokens).to(
                clip_model.text_projection.data.device
            )
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


def get_text_features(clip_model, class_names, prompt_creator):
    text_features = prompt_creator(
        class_names, clip_model
    )
    return text_features
