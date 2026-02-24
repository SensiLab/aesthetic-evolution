
import argparse
import types
from typing import Callable, Sequence

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


class CLIP_IQA:
    def __init__( 
        self,
        model_name: str = "RN50",
        pretrained_tag: str = "openai") -> None:


        self.model_name = model_name
        self.pretrained_tag = pretrained_tag
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained_tag,
            device=self.device,
        )
        self.model = self.remove_vision_positional_encoding(self.model)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def preprocess_image_like_open_clip(
        self,
        image: Image.Image,
        preprocess: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:
        """Apply OpenCLIP preprocessing while skipping resize and center crop."""

        if hasattr(preprocess, "transforms"):
            output = image
            for transform in preprocess.transforms:
                transform_name = transform.__class__.__name__
                if transform_name in {"Resize", "CenterCrop"}:
                    continue
                output = transform(output)
            return output

        return preprocess(image)

    def remove_vision_positional_encoding(self, model: torch.nn.Module) -> torch.nn.Module:
        """Remove vision positional encoding in RN50-style OpenCLIP attention pooling.

        This function modifies only the visual branch by monkey-patching
        ``model.visual.attnpool.forward`` (when present), leaving the text encoder and
        all other model components unchanged.

        Why this is needed:
        - In CLIP RN50, the attention-pooling layer adds a learned positional embedding
            whose length is tied to the training-time spatial grid (typically 224x224
            inputs after preprocessing).
        - If resize/crop are skipped and the input resolution changes, the token count
            no longer matches the positional-embedding length, which causes a shape
            mismatch.

        How it works technically:
        - It replaces ``attnpool.forward`` with a method that mirrors the original
            attention-pooling flow but omits positional-embedding addition.
        - The replacement still flattens spatial features to tokens, prepends a global
            mean token, and runs multi-head attention with the original
            ``q_proj/k_proj/v_proj/c_proj`` weights and biases.
        - Because no fixed-length positional vector is added, the number of spatial
            tokens can vary with input size, enabling arbitrary-resolution images.

        Notes:
        - The patch is a no-op for model variants without ``visual.attnpool`` or the
            expected projection attributes.
        - This changes visual semantics (no absolute positional cue), so scores may
            differ from the standard CLIP preprocessing path.
        """

        visual = getattr(model, "visual", None)
        attnpool = getattr(visual, "attnpool", None)

        if attnpool is None:
            return model

        required_attrs = [
            "q_proj",
            "k_proj",
            "v_proj",
            "c_proj",
            "num_heads",
        ]
        if not all(hasattr(attnpool, attr) for attr in required_attrs):
            return model

        def _forward_without_positional_embedding(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Attention-pool vision tokens without adding positional embeddings."""
            x = x.flatten(start_dim=2).permute(2, 0, 1)
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

            x, _ = F.multi_head_attention_forward(
                query=x[:1],
                key=x,
                value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False,
            )

            return x.squeeze(0)

        attnpool.forward = types.MethodType(_forward_without_positional_embedding, attnpool)
        return model

    def compute_clip_iqa_score(
        self,
        image_path: str | Sequence[str],
        positive_prompt: str = "Good photo.",
        negative_prompt: str = "Bad photo.") -> tuple[float, float] | list[tuple[float, float]]:

        image_paths: list[str]
        if isinstance(image_path, str):
            image_paths = [image_path]
        else:
            image_paths = list(image_path)

        if not image_paths:
            raise ValueError("image_path must contain at least one path")

        image_tensors = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image_tensors.append(self.preprocess_image_like_open_clip(image, self.preprocess))

        text_tokens = self.tokenizer([positive_prompt, negative_prompt]).to(self.device)

        with torch.no_grad():
            first_shape = image_tensors[0].shape
            same_shape = all(tensor.shape == first_shape for tensor in image_tensors)

            if same_shape:
                image_batch = torch.stack(image_tensors, dim=0).to(self.device)
                image_features = self.model.encode_image(image_batch)
            else:
                image_features_list = []
                for tensor in image_tensors:
                    single_feature = self.model.encode_image(tensor.unsqueeze(0).to(self.device))
                    image_features_list.append(single_feature)
                image_features = torch.cat(image_features_list, dim=0)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            logits = logits * self.model.logit_scale.exp()  # adjust by learned logit scale
            probs = logits.softmax(dim=-1)

        scores = [(probs[i, 0].item(), probs[i, 1].item()) for i in range(probs.shape[0])]
        if isinstance(image_path, str):
            return scores[0]
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP-IQA score with RN50 and no positional encoding in attnpool.")
    parser.add_argument("image", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--model", default="RN50", help="open_clip model name (default: RN50)")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag (default: openai)")
    parser.add_argument("--pos-prompt", default="Good Harmonograph", help="Positive quality prompt")
    parser.add_argument("--neg-prompt", default="Messy Harmonograph", help="Negative quality prompt")
    args = parser.parse_args()

    clip_iqa = CLIP_IQA(
        model_name=args.model,
        pretrained_tag=args.pretrained,
    )

    scores = clip_iqa.compute_clip_iqa_score(
        image_path=args.image,
        positive_prompt=args.pos_prompt,
        negative_prompt=args.neg_prompt,
    )

    if len(args.image) == 1:
        p_good, p_bad = scores[0]
        print(f"P(good): {p_good:.6f}")
        print(f"P(bad):  {p_bad:.6f}")
    else:
        for path, (p_good, p_bad) in zip(args.image, scores):
            print(path)
            print(f"  P(good): {p_good:.6f}")
            print(f"  P(bad):  {p_bad:.6f}")


if __name__ == "__main__":
    main()