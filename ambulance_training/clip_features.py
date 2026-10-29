"""CLIP ViT-B/16 (DataComp-XL) image embeddings + zero-shot vehicle-type probabilities (same prompts as the
Kaggle pre-sort job). Used for pre-sorting crops and as the feature extractor of the ambulance classifier."""
import numpy as np
import open_clip
import torch
from PIL import Image

CLASSES = {
    "ambulance": ["a photo of an ambulance", "a photo of an Indian ambulance van", "a photo of a 108 emergency ambulance"],
    "police": ["a photo of a police car", "a photo of a police jeep or police van"],
    "fire": ["a photo of a fire truck", "a photo of a fire engine"],
    "car": ["a photo of a car", "a photo of an SUV"],
    "van": ["a photo of a white van", "a photo of a minibus or passenger van", "a photo of a school van"],
    "bus": ["a photo of a bus"],
    "truck": ["a photo of a truck", "a photo of a pickup truck", "a photo of a water tanker truck"],
    "auto": ["a photo of an auto rickshaw"],
    "twowheeler": ["a photo of a motorcycle or scooter"],
    "other": ["a photo of a poster or text", "a photo of people", "a photo of a building", "a photo of a road without vehicles"],
}


class Clip:
    def __init__(self, arch="ViT-B-16", pretrained="datacomp_xl_s13b_b90k"):
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.prep = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=self.dev)
        self.model.eval()
        tok = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            T = []
            for ps in CLASSES.values():
                e = self.model.encode_text(tok(ps).to(self.dev)).float()
                e = e / e.norm(dim=-1, keepdim=True)
                T.append(e.mean(0))
            T = torch.stack(T)
            self.T = T / T.norm(dim=-1, keepdim=True)

    def embed(self, images, bs=64):
        """images: list of paths or BGR numpy arrays -> (n, 512) unit embeddings"""
        out = []
        for i in range(0, len(images), bs):
            batch = []
            for im in images[i:i + bs]:
                if isinstance(im, str):
                    im = Image.open(im).convert("RGB")
                else:
                    im = Image.fromarray(im[..., ::-1])
                batch.append(self.prep(im))
            if not batch:
                continue
            with torch.no_grad():
                e = self.model.encode_image(torch.stack(batch).to(self.dev)).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def score(self, images):
        E = self.embed(images)
        pr = torch.from_numpy(100 * E @ self.T.cpu().numpy().T).softmax(-1).numpy()
        return [{k: round(float(v), 4) for k, v in zip(CLASSES, p)} for p in pr], E.astype(np.float16)


class EmergencyProbe:
    """ambulance / police / fire / other from CLIP embeddings (train_probe.py); score = P(any emergency type)."""

    def __init__(self, path):
        z = np.load(path)
        self.W, self.b, self.classes = z["coef"], z["intercept"], [str(c) for c in z["classes"]]
        self.clip = Clip()

    def predict(self, images):
        E = self.clip.embed(images)
        L = E @ self.W.T + self.b
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        return P      # columns: self.classes
