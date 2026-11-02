# Measured retrieval run

Model: `b32-openai` · device: `cpu` · split: `test`
Direction: text-to-image · ensemble: False · queries: 54 · bootstrap groups: 6

When using the bundled shape dataset, these results are a compositional diagnostic, not a natural-image benchmark. Dataset scope must be considered before generalizing.

| Metric | Mean | Cluster bootstrap 95% CI |
|---|---:|---:|
| hit@1 | 0.2778 | [0.1296, 0.4074] |
| hit@5 | 0.8333 | [0.6481, 1.0000] |
| recall@10 | 0.9306 | [0.8102, 1.0000] |
| mrr | 0.5020 | [0.3504, 0.6358] |
| map@10 | 0.4923 | [0.3373, 0.6358] |
| ndcg@10 | 0.6017 | [0.4760, 0.7274] |

Warm latency p50/p95: 27.71/30.83 ms. Scope: warm text encoding + exact ranking.

## Lowest-scoring queries

- `green-square-left-2-exact`: two green squares on the left side of a plain background — top result `green-square-right-1`, relevant `green-square-left-2`; nDCG@10=0.000
- `green-square-left-2-paraphrase`: a plain canvas with two green squares placed to the left — top result `green-circle-left-2`, relevant `green-square-left-2`; nDCG@10=0.000
- `green-square-right-2-paraphrase`: a plain canvas with two green squares placed to the right — top result `green-square-right-1`, relevant `green-square-right-2`; nDCG@10=0.000
- `green-square-right-2-exact`: two green squares on the right side of a plain background — top result `green-square-right-1`, relevant `green-square-right-2`; nDCG@10=0.289
- `red-square-left-2-paraphrase`: a plain canvas with two red squares placed to the left — top result `red-triangle-left-1`, relevant `red-square-left-2`; nDCG@10=0.301
- `red-square-right-2-paraphrase`: a plain canvas with two red squares placed to the right — top result `red-square-right-1`, relevant `red-square-right-2`; nDCG@10=0.301
- `red-square-left-1-paraphrase`: a plain canvas with one red square placed to the left — top result `red-triangle-left-1`, relevant `red-square-left-1`; nDCG@10=0.333
- `red-square-left-2-exact`: two red squares on the left side of a plain background — top result `red-square-right-1`, relevant `red-square-left-2`; nDCG@10=0.333

Exact rankings, package versions, dataset/query hashes and prompt templates are in the adjacent JSON file.
