# ApexOracle-Evo2 维护说明

- 本仓库以 ArcInstitute Evo 2 upstream 为基础，保留 upstream 的 Apache-2.0 `LICENSE` 与 `NOTICE`。
- ApexOracle genome embedding 的 canonical CLI 为 `apexoracle-evo2-extract`，实现位于
  `apexoracle_evo2/`。它接受一个 FASTA 文件或目录，默认使用 11,000 nt window、10,000 nt step、
  Evo-2-40B `blocks.46.mlp.l3` activation 和有效 token mean pooling。
- 主要参数为 `--input`、`--output-dir`、`--model-name`、`--layer-name`、`--chunk-length`、
  `--step-length`、`--batch-size` 与 `--input-device`。`--plan-only` 只核验 FASTA/window contract，
  不加载模型；`--plan-detail files` 可按需输出逐文件明细。
- 每个输入 FASTA 输出 `<stem>.pt` 和 `<stem>.manifest.json`；manifest 记录输入 SHA-256、模型、层、
  window 参数、tensor shape/dtype，以及每个 window 的 record/index/start/end 血缘。
- FASTA 文件、checkpoint、tensor、cache、日志和实验输出不得进入 Git。发布前运行：
  `python -m pytest -q tests`、`python -m build`、`git diff --check`。
- `.github/workflows/apexoracle.yml` 在 Python 3.11/3.12 CPU 环境运行同一 focused tests 并构建
  source/wheel archives；40B GPU smoke 仍是独立的 release gate。
- 2026-08-10 Evo-2-40B release smoke 已使用 `vtx==1.1.0`、两张 H100 和正式缓存权重通过：合成
  two-record FASTA 产生 7 个有序 windows，tensor 为 `[7,8192]` / `torch.bfloat16`，7/7 rows 非零且
  全部 finite；manifest coordinates、partial flags、shape/dtype 与 tensor SHA-256 均通过复核。
- 对 upstream 文件的修改保持最小化；不得向 `ArcInstitute/evo2` 的 remote 推送 ApexOracle commit。
