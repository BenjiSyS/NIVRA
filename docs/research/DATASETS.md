# Candidate training data (for owner approval)

Researched 2026-10-21. **Nothing listed here has been downloaded.** Each source needs the owner's explicit
approval, and every ingested folder needs a `PROVENANCE.json` (see `docs/training/DATA_PIPELINE.md`).
A dataset's own licence (often for its *annotations*) does not clear the copyright of the underlying
videos; the per-clip source licence is what matters. Not legal advice — confirm terms before use.

## Tier A — clearest rights (recommended starting points)

| Source | What | Licence situation | Notes |
|--------|------|-------------------|-------|
| **Your own footage** | Videos you or contributors filmed, with written consent from identifiable people | You own it | Best for quality control; captions written by you |
| **Prelinger Archives** (Internet Archive) | Ephemeral films 1927–1987 (ads, educational, industrial) | Large share public domain; item metadata varies in reliability | Verify per item; old footage, limited modern content ([Wikipedia](https://en.wikipedia.org/wiki/Prelinger_Archives), [Public Domain Review](https://publicdomainreview.org/collections/source/prelinger-archives/)) |
| **NASA Image and Video Library** | Space/science video | Generally US-government public domain (check per item; some third-party material) | Needs captions from us |
| **Wikimedia Commons video** | Mixed footage | Per-file CC0/CC-BY/CC-BY-SA; attribution required for BY | Filter to CC0/CC-BY; SA terms may affect weight licensing |
| **CC-BY YouTube subset (via YouTube-Commons / Common Pile metadata)** | Openly licensed YouTube videos | CC-BY flag set by uploaders; known to be imperfect | Those datasets contain transcripts, not visual captions ([YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons), [Common Pile](https://arxiv.org/pdf/2506.05209)) |
| **TRECVID-VTT CC subset** | 2,710 short CC-licensed clips with 2–5 human captions each | Creative Commons (Flickr/V3C2) | Small but human-captioned; good for evaluation ([TRECVID 2020](https://arxiv.org/pdf/2104.13473)) |

## Tier B — popular, but underlying video rights are unclear

| Dataset | Notes |
|---------|-------|
| **OpenVid-1M** | Annotations CC BY 4.0; videos come from Panda-70M, ChronoMagic, Open-Sora-plan and CelebV-HQ (one source licence unknown) ([paper](https://arxiv.org/pdf/2407.02371)) |
| **VidGen-1M** | Curated from Panda-70M (YouTube-sourced); repository header suggests Apache-2.0 for the release, but that does not cover the videos ([paper](https://arxiv.org/html/2408.02629v1)) |
| **UltraVideo** | UHD-4K clips with captions; licence not verified ([paper](https://arxiv.org/html/2506.13691v1)) |

Recommendation: only consider Tier B after legal review, and never redistribute the videos.

## Not recommended

* **Pexels**: its terms prohibit automated collection for machine-learning purposes without explicit permission ([Pexels help](https://help.pexels.com/hc/de-de/articles/900005880463)).
* **Panda-70M, HD-VG-130M, WebVid**: YouTube/stock-sourced without clear training rights (WebVid was withdrawn).

## Text for the original Nivra text encoder

| Source | Notes |
|--------|-------|
| **Common Pile v0.1** | ~8 TB of public-domain and openly licensed text ([paper](https://arxiv.org/pdf/2506.05209)) — suitable for masked-language pretraining |
| **Common Corpus** | Large multilingual openly licensed corpus (includes Swedish) ([paper](https://arxiv.org/pdf/2506.01732)) |
| **WikiCaps** | 3.8 M image captions from Wikimedia Commons, text CC BY-SA 4.0 ([site](https://www.cl.uni-heidelberg.de/statnlpgroup/wikicaps/)) — caption-style language |

## Captions

Most Tier A video has no visual captions. Options: write them (best for quality and rights), or use a
captioning model. A third-party captioning model would be a separately licensed tool recorded in the
dataset provenance; it is not part of Nivra's weights.

## Content rules applied at ingestion

No sexual content involving minors under any circumstances; no sexual or deceptive content of real,
identifiable people without documented consent (`contains_people` / `consent` fields in `PROVENANCE.json`).
