# ApexOracle embedding re-extraction handoff

This document specifies one job: regenerate **all** ApexOracle conditioning embeddings
from scratch, for bacteria and for viruses, using the record-aware extraction path in
this fork.

Nothing from the previous embedding release should be reused. The previous producer
used a cross-record global window counter, which silently dropped every FASTA record
after the first once the counter passed that record's length. Single-contig bacterial
genomes were unaffected in practice, but the indexing is wrong in general and it is
catastrophic for segmented viral genomes: an 8-segment influenza genome collapsed to a
single window covering only segment 1.

## 1. Two input sets, two different models

This is the single most important instruction in this document.

| Input set | Genomes | Model | Layer |
| --- | --- | --- | --- |
| Bacteria / fungi | 568 | **stock, non-fine-tuned Evo 2 40B** | `blocks.46.mlp.l3` (frozen default) |
| Viruses | 80 | **our fine-tuned Evo 2 20B** | see section 4 — not yet decided |

The two sets are **not** in the same representation space and must never be mixed
inside one model input. They are consumed by separate downstream models: the bacterial
embeddings feed AMP MIC regression and small-molecule antibiotic classification, and the
viral embeddings feed a separate antiviral-peptide regression model.

Keep the two output directories separate. Do not merge the manifests.

> **TO BE FILLED BEFORE THE RUN:** the fine-tuned Evo 2 20B checkpoint location and the
> `--model-name` value it should be loaded under. The checkpoint is not in this
> repository and is not on the H100 host.

## 2. Code and version

```bash
git clone https://github.com/DragonDescentZerotsu/ApexOracle-Evo2.git
cd ApexOracle-Evo2
git checkout codex/fix-multi-contig-windowing
pip install -e .
```

This fork is `0.6.0+apexoracle.1`, based on ArcInstitute Evo 2 upstream commit `53f1959`.

Run the test suite before extracting anything:

```bash
python -m pytest tests/ -q       # expect 10 passed
```

## 3. The windowing contract

Every manifest this CLI writes contains:

```json
"window_indexing_contract": "per_record_zero_based_v1"
```

**Verify this string is present in every output manifest.** If it is absent, you are
running the wrong code or the wrong branch, and the resulting tensors must be discarded.

Window coordinates restart at zero for every FASTA record. 9 of the 80 viral genomes are
segmented and depend on this:

| Genome | Segments |
| --- | --- |
| rotavirus A | 11 |
| influenza A virus | 8 |
| influenza B virus | 8 |
| Rift Valley fever virus | 3 |
| Sin Nombre virus | 3 |
| Junín virus | 2 |
| Pichindé virus | 2 |
| Tacaribe virus | 2 |
| red-spotted grouper nervous necrosis virus | 2 |

Under the old indexing each of these produced exactly one window. Under this contract
they produce one window per segment (more for segments over 11,000 nt).

## 4. Layer selection

For the 40B bacterial run, use the frozen default `blocks.46.mlp.l3`. Pass nothing; the
CLI resolves it.

For the 20B viral run there is **no frozen default and no upstream recommendation**. We
checked both sources:

- The ArcInstitute Evo 2 README gives exactly one example, `blocks.28.mlp.l3`, and it is
  for the 7B model. It states only that intermediate embeddings work better than final
  embeddings.
- The NVIDIA NIM Evo 2 documentation covers 7B and 40B, explicitly declines to give a
  default, and says only to use an intermediate layer or a layer's final MLP output.

So the layer is a judgement call. Relative depth of the two known choices:

| Model | Layers | Hidden size | Layer used | Relative depth |
| --- | --- | --- | --- | --- |
| Evo 2 7B | 32 | 4096 | `blocks.28.mlp.l3` | 87.5% |
| Evo 2 40B | 50 | 8192 | `blocks.46.mlp.l3` | 92% |
| **Evo 2 20B** | **24** | **8192** | **undecided** | — |

Candidates for the 20B are `blocks.21.mlp.l3` (87.5%, matching the only published
upstream example) and `blocks.22.mlp.l3` (92%, matching our 40B choice). The 20B and 40B
share hidden size 8192, so either choice is dimensionally compatible downstream.

**This choice is yours.** If you have capacity, extracting the 80 viral genomes is cheap
(3.3 Mb of sequence total) and a 20/21/22/23 sweep is affordable; otherwise pick one and
record it. The layer must be passed explicitly:

```bash
--layer-name blocks.21.mlp.l3
```

Without `--layer-name`, a model with no frozen default fails with a clear error rather
than guessing. That is intentional.

## 5. Commands

Always dry-run the window plan first; it needs no GPU and no model weights:

```bash
apexoracle-evo2-extract \
  --input  Genome/ATCC \
  --output-dir out/bacteria_40b \
  --plan-only --plan-detail files
```

Bacteria, stock 40B:

```bash
CUDA_VISIBLE_DEVICES=0,1 apexoracle-evo2-extract \
  --input  Genome/ATCC \
  --output-dir out/bacteria_40b \
  --model-name evo2_40b \
  --batch-size 3 \
  --input-device cuda:0
```

Viruses, fine-tuned 20B:

```bash
CUDA_VISIBLE_DEVICES=0,1 apexoracle-evo2-extract \
  --input  Genome/Virus \
  --output-dir out/virus_20b_ft \
  --model-name <FINE_TUNED_20B_NAME> \
  --layer-name blocks.21.mlp.l3 \
  --batch-size 3 \
  --input-device cuda:0
```

Do not change `--chunk-length` or `--step-length`. The ApexOracle contract is 11,000 nt
windows with a 10,000 nt step, and terminal short windows are retained. Do not pass
`--full-windows-only`; many viral segments are under 11,000 nt and would vanish.

## 6. Inputs you receive

```
Genome/ATCC/<stem>.fasta                        568 bacterial genomes
Genome/Virus/<stem>.fasta                        80 viral genomes
Text_Description/Virus/Text/<stem>.txt           80 viral descriptions
manifests/virus_genome_manifest.tsv              per-genome provenance
manifests/target_to_genome.tsv                   DRAVP target -> genome
SHA256SUMS                                       checksums for everything above
```

Verify before starting:

```bash
sha256sum -c SHA256SUMS
```

Viral filenames use the ApexOracle `text-only` encoding, where `～` stands for a space
and `^` for a forward slash, because names such as `influenza A virus A/PR/8/34` cannot
be stored literally. Treat the stems as opaque; do not normalize them. The genome file
and its description file always share a stem.

## 7. Expected outputs

Per input FASTA, one tensor plus one JSON manifest:

- tensor: `float32` or `bfloat16`, shape `[n_windows, hidden_size]`, pooled by
  `valid_token_mean` over non-padding tokens
- manifest: FASTA path and SHA-256, model name, layer name,
  `window_indexing_contract`, `chunk_length`, `step_length`, `record_count`,
  `window_count`, tensor SHA-256 and shape

Return both directories in full, plus the console logs.

## 8. Text-modality embeddings

The ApexOracle strain encoder also consumes a text description per genome. The 80 viral
descriptions ship with this package. They follow the same four-section layout as the
existing bacterial descriptions, so the existing producer works unchanged.

This part does **not** use Evo 2. The contract is:

- model `YBXL/Med-LLaMA3-8B`, revision `567e7e71d8b6b433d8bc494f8112176bec4afccf`
- take hidden state index `-2` (penultimate)
- before encoding, the virus name decoded from the filename stem is replaced with the
  literal string `This strain`
- save a token-by-feature `float32` tensor

The helper CLI for this lives in the ApexOracle repository
(`scripts/prepare_data/embed_strain_texts.py`), not in this one. Tell us if you would
rather we run this step ourselves; it needs no large GPU.

## 9. Acceptance checks we will run

1. `window_indexing_contract == "per_record_zero_based_v1"` in every manifest.
2. `record_count` in each manifest equals the FASTA record count we shipped.
3. Tensor first dimension equals `window_count`, and `window_count` equals the count our
   own `--plan-only` run produces for the same input.
4. The 9 segmented viral genomes have `window_count >= segment_count`, and specifically
   influenza A returns 8 windows rather than 1.
5. Bacterial and viral tensors are in separate directories with separate manifests, and
   the model name recorded in each matches section 1.
