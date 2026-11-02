# Measured retrieval run

Model: `b32-openai` · device: `cpu` · split: `test`
Direction: image-to-text · ensemble: False · queries: 24 · bootstrap groups: 6

When using the bundled shape dataset, these results are a compositional diagnostic, not a natural-image benchmark. Dataset scope must be considered before generalizing.

| Metric | Mean | Cluster bootstrap 95% CI |
|---|---:|---:|
| hit@1 | 0.3333 | [0.2500, 0.4167] |
| hit@5 | 0.9583 | [0.8750, 1.0000] |
| recall@10 | 1.0000 | [1.0000, 1.0000] |
| mrr | 0.5754 | [0.5119, 0.6424] |
| map@10 | 0.5754 | [0.5119, 0.6424] |
| ndcg@10 | 0.6809 | [0.6323, 0.7324] |

Warm latency p50/p95: 0.03/0.06 ms. Scope: ranking only; image/caption embeddings cached.

## Lowest-scoring queries

- `red-square-left-1-image-query`: one red square on the left side of a plain background — top result `red-square-right-2`, relevant `red-square-left-1`; nDCG@10=0.333
- `blue-circle-left-1-image-query`: one blue circle on the left side of a plain background — top result `blue-circle-right-2`, relevant `blue-circle-left-1`; nDCG@10=0.431
- `blue-triangle-left-1-image-query`: one blue triangle on the left side of a plain background — top result `blue-triangle-right-2`, relevant `blue-triangle-left-1`; nDCG@10=0.431
- `green-square-left-1-image-query`: one green square on the left side of a plain background — top result `green-square-right-1`, relevant `green-square-left-1`; nDCG@10=0.431
- `yellow-triangle-left-1-image-query`: one yellow triangle on the left side of a plain background — top result `yellow-triangle-right-2`, relevant `yellow-triangle-left-1`; nDCG@10=0.431
- `red-square-right-1-image-query`: one red square on the right side of a plain background — top result `red-square-right-2`, relevant `red-square-right-1`; nDCG@10=0.500
- `blue-circle-right-1-image-query`: one blue circle on the right side of a plain background — top result `blue-circle-right-2`, relevant `blue-circle-right-1`; nDCG@10=0.500
- `blue-triangle-right-1-image-query`: one blue triangle on the right side of a plain background — top result `blue-triangle-left-2`, relevant `blue-triangle-right-1`; nDCG@10=0.500

Exact rankings, package versions, dataset/query hashes and prompt templates are in the adjacent JSON file.
