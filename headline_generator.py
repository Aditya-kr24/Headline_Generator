"""Grounded headline generation for images of public events.

Pipeline:
  image -> BLIP caption -> CLIP event-type guess -> OCR of visible text
        -> evidence bundle -> instruction-tuned LLM -> grounding check -> headline

The grounding check rejects headlines containing proper nouns or numbers that
are not present in the evidence (caption / OCR / event label), and falls back
to a caption-based headline, so facts that are not visible are not invented.
"""
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
CLIP_MODEL = "openai/clip-vit-base-patch32"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

EVENT_LABELS = [
    "a political rally", "a protest march", "a religious festival procession",
    "a music concert", "a sports match", "a running race",
    "a street market", "a press conference", "a public speech on a stage",
    "a cultural dance performance", "a parade", "a graduation ceremony",
    "an exhibition or fair", "a candlelight vigil", "a crowd gathered in a public square",
]
CLIP_CONFIDENCE = 0.35

SYSTEM_PROMPT = (
    "You are a careful news-desk editor writing headlines for photographs. "
    "Use ONLY the evidence provided. Never add names, places, dates, numbers, "
    "causes, or opinions that are not in the evidence. If the evidence is vague, "
    "keep the headline general. Write one headline in sentence case, "
    "6 to 14 words, present tense, no quotation marks, no trailing period."
)

STOPWORDS_OK = {"a", "an", "the", "of", "in", "at", "on", "and", "with", "for", "as"}


@dataclass
class Result:
    caption: str
    event_type: Optional[str]
    event_confidence: float
    ocr_text: str
    headline: str
    grounded: bool
    used_fallback: bool

    def to_dict(self):
        return asdict(self)


class HeadlineGenerator:
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        self._blip = self._blip_proc = None
        self._clip = self._clip_proc = None
        self._llm = self._tok = None
        self._ocr = None

    # ---------- lazy model loading ----------
    def _load_blip(self):
        if self._blip is None:
            self._blip_proc = BlipProcessor.from_pretrained(CAPTION_MODEL)
            self._blip = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(DEVICE).eval()

    def _load_clip(self):
        if self._clip is None:
            self._clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
            self._clip = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()

    def _load_llm(self):
        if self._llm is None:
            self._tok = AutoTokenizer.from_pretrained(LLM_MODEL)
            self._llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            ).to(DEVICE).eval()

    def _load_ocr(self):
        if self._ocr is None:
            import easyocr  # imported lazily; OCR is optional
            self._ocr = easyocr.Reader(["en"], gpu=DEVICE == "cuda")

    # ---------- pipeline stages ----------
    @torch.no_grad()
    def caption(self, image: Image.Image) -> str:
        self._load_blip()
        inputs = self._blip_proc(image, return_tensors="pt").to(DEVICE)
        out = self._blip.generate(**inputs, num_beams=5, max_new_tokens=40)
        return self._blip_proc.decode(out[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def event_type(self, image: Image.Image):
        self._load_clip()
        inputs = self._clip_proc(
            text=[f"a photo of {l}" for l in EVENT_LABELS],
            images=image, return_tensors="pt", padding=True,
        ).to(DEVICE)
        probs = self._clip(**inputs).logits_per_image.softmax(dim=1)[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        return (EVENT_LABELS[idx], conf) if conf >= CLIP_CONFIDENCE else (None, conf)

    def read_text(self, image: Image.Image) -> str:
        if not self.use_ocr:
            return ""
        try:
            import numpy as np
            self._load_ocr()
            w, h = image.size
            big = image.convert("RGB").resize((w * 2, h * 2))  # upscale helps small banner text
            parts = self._ocr.readtext(np.array(big), detail=1)
            from spellchecker import SpellChecker  # drop OCR noise such as "GMERICAN"
            spell = SpellChecker()
            words = [t.strip() for _, t, c in parts
                     if c >= 0.4 and len(t.strip()) >= 3 and t.strip().isalpha()
                     and spell.known([t.strip().lower()])]
            return " ".join(dict.fromkeys(words)).strip()
        except Exception:
            return ""

    @staticmethod
    def build_evidence(caption, event, ocr_text) -> str:
        lines = [f"Scene caption: {caption}"]
        if event:
            lines.append(f"Likely event type: {event}")
        lines.append(f"Text visible in image: {ocr_text if ocr_text else 'none detected'}")
        return "\n".join(lines)

    @torch.no_grad()
    def _llm_headline(self, evidence: str, extra: str = "") -> str:
        self._load_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + extra},
            {"role": "user", "content": f"Evidence:\n{evidence}\n\nHeadline:"},
        ]
        prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(DEVICE)
        out = self._llm.generate(**inputs, max_new_tokens=32, do_sample=False)
        text = self._tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self.clean(text)

    # ---------- grounding ----------
    @staticmethod
    def clean(text: str) -> str:
        text = text.strip().split("\n")[0].strip().strip('"\'“”‘’')
        return text.rstrip(".").strip()

    @staticmethod
    def sentence_case(headline: str, ocr_text: str = "") -> str:
        """Fix Title Case / ALL CAPS: lower-case every word except those
        capitalised in the OCR text (real visible names), then capitalise the first letter."""
        ocr_words = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", ocr_text))
        out = []
        for w in headline.split():
            core = re.sub(r"[^A-Za-z0-9'\-]", "", w)
            out.append(w if core in ocr_words else w.lower())
        s = " ".join(out)
        return s[:1].upper() + s[1:] if s else s

    @staticmethod
    def ungrounded_terms(headline: str, evidence: str) -> List[str]:
        """Proper nouns (capitalised, not sentence-initial) and numbers that are
        absent from the evidence."""
        ev = evidence.lower()
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", headline)
        bad = []
        for i, w in enumerate(words):
            is_num = any(ch.isdigit() for ch in w)
            is_proper = i > 0 and w[0].isupper() and w.lower() not in STOPWORDS_OK
            if (is_num or is_proper) and w.lower() not in ev:
                bad.append(w)
        return bad

    @staticmethod
    def fallback_headline(caption: str, event: Optional[str]) -> str:
        cap = caption.strip().rstrip(".")
        cap = re.sub(r"^(there (is|are)|this is|here (is|are))\s+", "", cap, flags=re.I)
        return cap[0].upper() + cap[1:] if cap else "Crowd gathers at public event"

    # ---------- public API ----------
    def generate(self, image: Image.Image) -> Result:
        image = image.convert("RGB")
        cap = self.caption(image)
        event, conf = self.event_type(image)
        ocr_text = self.read_text(image)
        evidence = self.build_evidence(cap, event, ocr_text)

        headline = self.sentence_case(self._llm_headline(evidence), ocr_text)
        bad = self.ungrounded_terms(headline, evidence)
        used_fallback = False
        if bad:  # one constrained retry, then fall back
            headline = self.sentence_case(self._llm_headline(
                evidence,
                extra=f" Do NOT use these words, they are unsupported: {', '.join(bad)}.",
            ), ocr_text)
            bad = self.ungrounded_terms(headline, evidence)
        if bad or not headline:
            headline = self.fallback_headline(cap, event)
            used_fallback = True
        return Result(cap, event, conf, ocr_text, headline, not bad, used_fallback)
