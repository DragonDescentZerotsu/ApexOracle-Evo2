# ApexOracle genome embedding re-extraction

Everything needed to regenerate the ApexOracle conditioning embeddings is in this
directory. If you only read one page, read this one.

**The job:** run one extraction command over `genomes/bacteria/` and a second over
`genomes/virus/`, with a different model for each, and send back the tensors and
manifests.

This branch replaces the upstream Evo 2 landing page with these instructions. The
original ArcInstitute README is preserved as
[`UPSTREAM_README.md`](UPSTREAM_README.md), and `main` still carries it as the
repository README.

---

## 1. Why everything is being redone

The previous extraction did not reset its window counter between FASTA records.
Once the running counter passed a record's length, every remaining record in that
file produced no windows at all and was silently dropped.

This was believed to affect only segmented viral genomes. It does not. Measured
over the 568 bacterial and fungal genomes in this package:

| | Old indexing | Correct indexing |
| --- | --- | --- |
| Windows | 210,206 | 340,188 |
| Contigs covered | 568 of 3,390 | 3,390 of 3,390 |

**370 of 568 genomes lost whole contigs, 83% of all contigs were dropped, and
38% of the sequence never reached the model.** Fungal assemblies were worst:
*Aspergillus ustus* ATCC 1041 has 289 contigs and produced 113 windows instead of
4,133. Segmented viruses collapsed to a single window covering only segment 1.

No previous embedding can be reused, for any organism.

## 2. Install

```bash
git clone https://github.com/DragonDescentZerotsu/ApexOracle-Evo2.git
cd ApexOracle-Evo2
git checkout virus-extension
pip install -e .
python -m pytest tests/ -q          # expect 10 passed
```

The branch is `virus-extension`. The fix and its regression test live there; `main`
tracks upstream ArcInstitute Evo 2 and does not contain the extraction CLI.

## 3. Get the data

The genomes are a private dataset on the Hub. If you get a 404, ask us to add
your Hub account; the repository is
[`Kiria-Nozan/apexoracle-genome-handoff`](https://huggingface.co/datasets/Kiria-Nozan/apexoracle-genome-handoff).

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download Kiria-Nozan/apexoracle-genome-handoff \
  --repo-type dataset --local-dir apexoracle-genomes
cd apexoracle-genomes
```

About 3.3 GB, almost all of it the bacterial and fungal FASTAs. The download
contains:

```
genomes/bacteria/<name>.fasta        568 bacterial and fungal genomes
genomes/virus/<name>.fasta            80 viral genomes
text/virus/<name>.txt                 80 viral descriptions   (section 7)
manifests/virus_genome_manifest.tsv   per-genome provenance
manifests/target_to_genome.tsv        DRAVP target -> genome file
SHA256SUMS                            checksums for everything above
```

Verify first:

```bash
sha256sum -c SHA256SUMS
```

Viral filenames encode a space as `～` and a slash as `^`, because names such as
`influenza A virus A/PR/8/34` cannot be stored literally. Treat the stems as
opaque and do not normalise them. A genome file and its description file always
share a stem.

## 4. Two input sets, two different models

This is the one thing that must not be mixed up.

| Input set | Genomes | Model | Layer |
| --- | --- | --- | --- |
| `genomes/bacteria/` | 568 | stock, **non-fine-tuned Evo 2 40B** | `blocks.46.mlp.l3` (frozen default) |
| `genomes/virus/` | 80 | **our fine-tuned Evo 2 20B** | you choose, see section 6 |

The two are different representation spaces and feed separate downstream models.
Keep the outputs in separate directories and do not merge the manifests.

> **The viral checkpoint is yours.** It is the Evo 2 20B viral LoRA you trained
> from the `Evo2_virus` handoff, so we do not ship it. Note that the handoff
> README originally requested a 40B base and the run was actually done on 20B;
> 20B is the one we want. Load the artifact from that run and record the exact
> checkpoint identity and the `--model-name` you used, because the manifest is
> the only thing tying these tensors back to a specific training run.

## 5. Commands

Dry-run the window plan first. It needs no GPU and no model weights, and it is
the cheapest way to confirm your copy of the data matches ours:

```bash
apexoracle-evo2-extract \
  --input genomes/bacteria \
  --output-dir out/bacteria_40b \
  --plan-only --plan-detail files
```

Bacteria and fungi, stock 40B:

```bash
CUDA_VISIBLE_DEVICES=0,1 apexoracle-evo2-extract \
  --input genomes/bacteria \
  --output-dir out/bacteria_40b \
  --model-name evo2_40b \
  --batch-size 3 \
  --input-device cuda:0
```

Viruses, fine-tuned 20B:

```bash
CUDA_VISIBLE_DEVICES=0,1 apexoracle-evo2-extract \
  --input genomes/virus \
  --output-dir out/virus_20b_ft \
  --model-name <FINE_TUNED_20B_NAME> \
  --layer-name blocks.21.mlp.l3 \
  --batch-size 3 \
  --input-device cuda:0
```

Do not change `--chunk-length` or `--step-length`: the contract is 11,000 nt
windows with a 10,000 nt step. Do not pass `--full-windows-only` — many viral
segments are shorter than 11,000 nt and would vanish.

Expect roughly 340,000 windows for the bacterial set and about 400 for the viral
set, so the bacterial run dominates the cost.

## 6. Choosing the 20B layer

There is no upstream recommendation. We checked: the ArcInstitute README gives one
example, `blocks.28.mlp.l3`, and it is for the 7B; the NVIDIA NIM documentation
explicitly declines to give a default. So this is a judgement call.

| Model | Blocks | Hidden | Layer | Relative depth |
| --- | --- | --- | --- | --- |
| Evo 2 7B | 32 | 4096 | `blocks.28.mlp.l3` | 87.5% |
| Evo 2 40B | 50 | 8192 | `blocks.46.mlp.l3` | 92% |
| **Evo 2 20B** | **24** | **8192** | **your choice** | — |

`blocks.21.mlp.l3` matches the only published upstream example at 87.5%;
`blocks.22.mlp.l3` matches our 40B choice at 92%. The 20B and 40B share hidden
size 8192, so either is dimensionally compatible downstream.

The viral set is small — 3.4 MB of sequence, about 400 windows — so a sweep over
blocks 20 through 23 is cheap if you have the capacity. Otherwise pick one and
record it. It must be passed explicitly; without `--layer-name` a model with no
frozen default fails with a clear error rather than guessing.

## 7. Text descriptions

The strain encoder also consumes one text description per genome, and the 80
viral ones are in `text/virus/`. This part does **not** use Evo 2:

- model `YBXL/Med-LLaMA3-8B`, revision `567e7e71d8b6b433d8bc494f8112176bec4afccf`
- hidden state index `-2` (penultimate)
- the virus name decoded from the filename stem is replaced with the literal
  string `This strain` before encoding
- save a token-by-feature `float32` tensor

The helper CLI lives in the ApexOracle repository, not the Evo 2 one. Tell us if
you would rather we run this step; it needs no large GPU.

## 8. What to send back

Per input FASTA, one tensor and one JSON manifest:

- tensor, shape `[n_windows, hidden_size]`, pooled by `valid_token_mean` over
  non-padding tokens
- manifest carrying FASTA path and SHA-256, model name, layer name,
  `window_indexing_contract`, `chunk_length`, `step_length`, `record_count`,
  `window_count`, tensor SHA-256 and shape

Send both output directories in full, plus the console logs.

## 9. What we check on arrival

1. `window_indexing_contract == "per_record_zero_based_v1"` in every manifest.
   If that string is missing you ran the wrong branch and the tensors are void.
2. `record_count` matches the FASTA record count we shipped.
3. Tensor first dimension equals `window_count`, and `window_count` matches our
   own `--plan-only` run. Bacterial total should be about 340,188, not 210,206.
4. Segmented genomes return one window per segment: influenza A must give 8, not 1.
5. Bacterial and viral outputs are in separate directories, with the model name
   in each manifest matching section 4.
6. **Activation scale.** The existing 40B tensors have a median `mean(abs(E))` of
   about `2.2e-15`, and ApexOracle compensates with a fixed `1e14` multiplier. We
   will recompute this for both new sets, because a fine-tuned model may not land
   on the same scale and the multiplier would then be wrong. Nothing for you to
   do beyond sending the tensors, but if you notice all-zero, NaN or inf tensors,
   say so rather than shipping them.
