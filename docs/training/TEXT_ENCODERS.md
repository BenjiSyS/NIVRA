# Text encoders

Contract: `docs/research/NIVRA_720P_DESIGN.md` section 3. Code: `ai/nivra/text/`.

```python
from nivra.text.registry import get_text_encoder
enc = get_text_encoder("nivra:runs/text/base")        # original Nivra weights
enc = get_text_encoder("external:D:/models/umt5-xl")   # separately licensed, opt-in
emb, mask = enc.encode(["en röd boll rullar", "a red ball"])   # [B,L,D] float32, [B,L] bool
enc.id, enc.dim, enc.max_len
```

## Nivra Text (original)

* `bpe.py`: byte-level BPE written from scratch. Ids: 0-3 specials (`<pad> <bos> <eos> <mask>`), 4-259 raw
  bytes, 260+ merges. Any UTF-8 string encodes and decodes exactly (no unknown token); Swedish, emoji and
  CJK are covered by the byte base. `tokenizer.json` stores version, merges and a sha256 that is verified on load.
* `nivra_text.py`: pre-norm transformer encoder, RoPE positions, final LayerNorm. Presets: `tiny` (tests),
  `small` (d=256, 6 layers), `base` (d=768, 12 layers, 12 heads, 3072 FFN; about 110 M parameters with a 32k vocab,
  checked by `count_params_for`). Checkpoint directory = `config.json` + `model.safetensors` + `tokenizer.json`;
  `config.json` records the tokenizer sha256 and the weights sha256, and loading refuses a mismatching tokenizer.
* Pretraining: `python -m nivra.cli text-train-tokenizer --input <files/dirs> --out tok.json --vocab-size 32000`, then
  `python -m nivra.cli text-pretrain --corpus <files/dirs> --tokenizer tok.json --out ckpt --size tiny|small|base --steps N`.
  BERT-style masked LM (15 %: 80/10/10). `--fp16` (CUDA autocast + GradScaler) is implemented but UNTESTED.
* Alignment hook for later joint training: `info_nce(text_pooled, video_emb)` (symmetric InfoNCE) and
  `ContrastiveHead` (projects pooled text and video embeddings into a shared space).
* Expectation: without a large licensed text corpus and caption alignment the encoder is weak. The tiny run in the
  tests only proves that the loop learns (loss falls), not that prompts are understood.

Speed (pure Python, measured on the dev laptop, i5-8250U): training a 2000-token vocabulary on a 135 KB corpus took 3.9 s;
encoding the same text about 1 MB/s with a cold per-word cache. Training time grows roughly with
vocab_size x distinct-word count, so a 32k vocabulary on a multi-GB corpus is a long, single-threaded job.

## External encoders (opt-in, third party)

`external.py` loads an encoder the owner already placed in a local directory. Rules:

* nothing is downloaded: URLs and UNC paths are refused, `local_files_only=True`, HF offline variables are set;
* `NIVRA_LICENSE.json` must be next to the weights with `name`, `license`, `source_url`, `sha256`; the sha256 must match the
  `*.safetensors` files (several files: sha256 of the sorted per-file digests joined by newlines); pickle formats are not accepted;
* id is `external:<name>@<sha256[:12]>`; `manifest_entry()` returns the record (origin `external-third-party`, license, URL)
  that model manifests must carry. It is never labelled as Nivra weights;
* building the network needs the optional `transformers` package (and `sentencepiece` for T5-family tokenizers). Neither is a
  Nivra dependency; without them a clear error says what to install.

## Originality and licensing

| Choice | Weights | Originality | Licence obligations |
|---|---|---|---|
| `nivra:` | trained by Nivra on a corpus you are licensed to use | fully original | the corpus licence only |
| `external:` | someone else's | the generator is original, the text understanding is not | that encoder's licence (attribution, use restrictions) travels with every checkpoint trained against it |

A generator trained with encoder X records X's id and refuses to load with another one. Use `nivra:` for a claim of
fully original weights; use `external:` only knowingly, with the licence file in place.
