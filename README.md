# AI Village Research

A local research hub for browsing the AI Village transcript, quantitative analyses, source-linked fieldnotes, and experimental DSPy message signatures.

## Run the website

Place `village-transcript.json` from the [AI Digest dataset](https://huggingface.co/datasets/aidigestorg/ai-village/blob/main/village-transcript.json) in the repository root, then run:

```sh
python3 viewer.py
```

Open [localhost:8765](http://127.0.0.1:8765/). The server uses Python's standard library; the pages use dependency-free JavaScript and shared CSS.

| View | URL |
| --- | --- |
| Research hub | `/` |
| Village transcript | `/village/` |
| Quantitative analysis | `/analysis.html` |
| Fieldnotes | `/fieldnotes/` |
| Message signatures | `/signatures/` |

The analyses use a frozen transcript containing 375,426 events across 404 day records. Source hashes distinguish that snapshot from newer downloads. Original dataset context and attribution are preserved in [DATASET_README.md](DATASET_README.md) and [SCHEMA.md](SCHEMA.md).

## Local models and signatures

The signature lab indexes 183,485 chat records and reuses predictions for identical text. It scores authored conversational-act hypotheses and proposes reusable signature templates. These are experimental classifications, not validated task extractions.

See [message_signatures/README.txt](message_signatures/README.txt) for environment setup, pinned model provenance, offline inference commands, pause/resume behavior, and report exports. Model weights, the virtual environment, and the runtime SQLite database are local downloads/build products and are not committed. The JSON pilot results and dated run checkpoint are included; the checkpoint is not a claim of full completion.

Apple Silicon can use the optional MLX backend with the same model and 512-token limit. A controlled 48-message benchmark measured 7.4× faster inference; a separate 68-message check preserved every top label, review flag, and truncation decision. See [speed measurements](message_signatures/acceleration-speed.json) and [parity results](message_signatures/acceleration-benchmark.json). These checks test implementation agreement, not classification accuracy or guaranteed full-run duration.

The analyzed snapshot is uploaded to [maujim/ai-village-analysis on Hugging Face](https://huggingface.co/datasets/maujim/ai-village-analysis) as a private dataset. It includes all 183,485 chat rows; the first upload contains annotations on 8,230 rows (7,344 unique texts). Pending annotations are explicitly null. The full local run continues separately; this uploaded snapshot is not automatically updated.

## Development

[PROJECT_README.txt](PROJECT_README.txt) documents routes, cache behavior, generated pages, and test commands. All model inference runs locally; the web viewer and the classification worker are independent processes.

The GitHub history contains the application and derived analysis payloads. The upstream Hugging Face dataset export history is kept separately.
