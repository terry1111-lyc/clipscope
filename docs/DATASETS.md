# 数据与评测合同

运行 `clipscope demo-data` 可生成原创合成诊断集。运行时的数据与索引在 `.gitignore` 中，不随代码仓库提交。生成器是公开数据来源，方便复现。

## 自定义图库

`manifest.jsonl` 每行描述一张图，路径相对于清单所在目录：

```json
{"id":"img001","path":"images/img001.jpg","caption":"a dog running on grass","tags":["animal"],"license":"your-source-license"}
```

- `id` 最长 128 字符，以 ASCII 字母或数字开头，仅允许字母、数字、点、下划线与短横线。
- `path` 只接受数据目录内现存的 PNG、JPEG、WebP；路径穿越和指向目录外的符号链接会被拒绝。
- 图片限制 1600 万像素；EXIF 方向会被应用，然后转为 RGB。
- `caption` 用于界面和图到文评测，不用于图像编码。真实图像集的 caption 应由独立来源标注。
- 相同文件内容会被拒绝；当前未实现感知哈希或语义近重复检测。
- 图库上限 20,000 张。资源不足时先用较小子集和 `--batch-size 2`。

`queries.jsonl` 每行描述一个查询：

```json
{"id":"q001","text":"a running dog","relevant":["img001"],"source_id":"img001","group":"source-event-01","split":"test","slice":"action"}
```

`relevant` 必须非空且全部在图库中，支持多个正确答案。`source_id` 关联图像，供图到文评测去重；`group` 是 bootstrap 和 split 检查的单位，同一组不能跨 dev/test。每个源图的改写至少应在同一组；高度相关的事件、对象系列也建议分到同一组。`slice` 表示预先定义的分析类别。

```bash
clipscope index --manifest data/custom/manifest.jsonl --index data/custom.sqlite
clipscope evaluate --index data/custom.sqlite --queries data/custom/queries.jsonl \
  --out experiments/results/custom-baseline.json
CLIPSCOPE_DATA=data/custom CLIPSCOPE_INDEX=data/custom.sqlite clipscope serve
```

使用不同模型时，index、evaluate 和服务的 `CLIPSCOPE_MODEL` 必须一致。

## 导入真实图像

脚本接受已经获得使用权并下载到本地的 COCO 格式 caption 注释与图片。项目不自动下载大数据集，不捆绑外部图片，也不替你确认每张图片的许可。

```bash
python scripts/import_coco.py \
  --annotations /path/to/captions_val2017.json \
  --images /path/to/val2017 \
  --out data/coco-subset --limit 1000 --seed 42
clipscope suite --manifest data/coco-subset/manifest.jsonl \
  --queries data/coco-subset/queries.jsonl --index data/coco.sqlite \
  --out experiments/coco-results
```

该脚本抽取最多 1000 张图，按源图划分 20% dev / 80% test，共用固定图库。它是本地子集实验，**不等于官方 COCO / Karpathy split**；不能与其排行榜数值直接比较。图到文只使用每图第一条 caption，其他 caption 仍用于文到图查询。

先在 dev 上确定模板、批大小等超参数，再冻结协议并运行 test。若要做标准公开基准，应另外实现并测试该基准的完整 candidate pool、多 caption 相关性与分割方式。
