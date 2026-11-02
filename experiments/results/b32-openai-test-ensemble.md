# Measured retrieval run

Model: `b32-openai` · device: `cpu` · split: `test`
Direction: text-to-image · ensemble: True · queries: 54 · bootstrap groups: 6

When using the bundled shape dataset, these results are a compositional diagnostic, not a natural-image benchmark. Dataset scope must be considered before generalizing.

| Metric | Mean | Cluster bootstrap 95% CI |
|---|---:|---:|
| hit@1 | 0.3148 | [0.2222, 0.4074] |
| hit@5 | 0.8333 | [0.6481, 1.0000] |
| recall@10 | 0.9306 | [0.8102, 1.0000] |
| mrr | 0.5350 | [0.4217, 0.6327] |
| map@10 | 0.5230 | [0.3957, 0.6327] |
| ndcg@10 | 0.6255 | [0.5237, 0.7250] |

Warm latency p50/p95: 50.57/55.21 ms. Scope: warm text encoding + exact ranking.

## Lowest-scoring queries

- `green-square-left-2-exact`: two green squares on the left side of a plain background — top result `green-square-right-1`, relevant `green-square-left-2`; nDCG@10=0.000
- `green-square-left-2-paraphrase`: a plain canvas with two green squares placed to the left — top result `green-square-right-1`, relevant `green-square-left-2`; nDCG@10=0.000
- `green-square-right-2-paraphrase`: a plain canvas with two green squares placed to the right — top result `green-square-right-1`, relevant `green-square-right-2`; nDCG@10=0.000
- `red-square-left-2-paraphrase`: a plain canvas with two red squares placed to the left — top result `red-square-right-1`, relevant `red-square-left-2`; nDCG@10=0.289
- `green-square-right-2-exact`: two green squares on the right side of a plain background — top result `green-square-right-1`, relevant `green-square-right-2`; nDCG@10=0.289
- `red-square-left-2-exact`: two red squares on the left side of a plain background — top result `red-square-right-1`, relevant `red-square-left-2`; nDCG@10=0.315
- `red-square-right-2-paraphrase`: a plain canvas with two red squares placed to the right — top result `red-square-right-1`, relevant `red-square-right-2`; nDCG@10=0.315
- `red-square-left-1-paraphrase`: a plain canvas with one red square placed to the left — top result `red-triangle-left-1`, relevant `red-square-left-1`; nDCG@10=0.333

Exact rankings, package versions, dataset/query hashes and prompt templates are in the adjacent JSON file.
