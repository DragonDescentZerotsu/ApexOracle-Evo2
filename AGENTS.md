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
- 对 upstream 文件的修改保持最小化；不得向 `ArcInstitute/evo2` 的 remote 推送 ApexOracle commit。
