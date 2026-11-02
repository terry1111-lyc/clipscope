# 运行与部署

## Apple Silicon 原生运行

使用 ARM64 Python 3.13 虚拟环境，安装 `.[model]`。`auto` 优先 CUDA，其次 MPS，最后 CPU；可通过 `CLIPSCOPE_DEVICE=cpu` 明确指定。查看实际设备，不根据硬件名称假定启用了 MPS：

```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
clipscope demo-data
clipscope index
clipscope serve
```

索引与实验 CLI 使用 `--device`；服务读取 `CLIPSCOPE_DEVICE`。本次运行的 MPS 检测返回 false，实测为 CPU。MPS 支持要求见 [PyTorch 官方说明](https://docs.pytorch.org/docs/stable/notes/mps.html)，不能将本报告延迟宣传为 M1 GPU 性能。

## Docker / Linux CPU

```bash
docker compose build
docker compose run --rm clipscope clipscope demo-data
docker compose run --rm clipscope clipscope index --device cpu
docker compose up -d
docker compose logs -f clipscope
```

镜像非 root 运行；数据、权重缓存在 named volume 中。首次初始化可能需要数分钟和数 GB 磁盘空间。容器只启动一个 worker，避免复制模型。Mac Docker 的 Linux 虚拟机不能使用 macOS MPS。

`docker compose run --rm clipscope clipscope suite --device cpu` 可在容器内运行评测，但默认结果目录不挂载，会随临时容器删除。需要保留时使用：

```bash
docker compose run --rm clipscope clipscope suite --device cpu --out data/reports
```

随后在 compose 服务环境中设置 `CLIPSCOPE_REPORTS: /app/data/reports` 并重建服务。随镜像附带的 `experiments/results` 是仓库中实际生成的报告，不代表容器硬件上重新测量。

当前环境没有 Docker，镜像未在本地构建验证；提供的 GitHub Actions 会构建镜像并执行 CLI smoke test。首次推送后应检查 CI 结果。

## 配置

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| CLIPSCOPE_MODEL | b32-openai | 与索引匹配的 checkpoint |
| CLIPSCOPE_DEVICE | auto | auto/cpu/mps/cuda |
| CLIPSCOPE_DATA | data/shapes | 索引中相对图片路径的根目录 |
| CLIPSCOPE_INDEX | data/index.sqlite | 只读服务索引 |
| CLIPSCOPE_CACHE | .cache/models | 权重缓存 |
| CLIPSCOPE_REPORTS | experiments/results | 展示的报告目录 |
| CLIPSCOPE_API_TOKEN | 空 | 设置后保护所有 /api/ 路径 |

服务不会自动读取 `.env`，shell 使用 export；Docker Compose 自动读取 `.env` 中的 token。UI 的 token 只存在当前页面内存，不写 localStorage。图片也通过带 Authorization 的请求加载。

## 对公网提供服务

默认只绑定 `127.0.0.1`，适合本地演示。需要公网演示时，在你选择的服务器上先启动同样的容器，再由 HTTPS 反向代理转发，并设置长随机 `CLIPSCOPE_API_TOKEN`。不要把 token 写入 GitHub。

使用上游代理设置每客户端速率限制、连接超时与并发上限；应用的 8 MiB 实际请求体限制无法防止大量并发请求占满内存。上传图片最大 1600 万像素，不持久化；索引更新仅由 CLI 完成。`/healthz`、静态 UI 与 OpenAPI 文档保持公开，不返回图片或模型权重。

本项目没有用户账号、队列、计费、多租户隔离、监控告警或多机扩展。部署前应基于实际访问量补齐相应设施。此次工作没有开通云资源，没有产生云 GPU 费用。

## 常见故障

- 下载失败：检查 DNS、公开模型源与网络连接，再重试；不要把随机权重当作预训练模型。
- 模型指纹不匹配：用当前模型/依赖重新 `index`，然后重启服务。
- 图片哈希变化：恢复原图片或重建索引；不要绕过检查。
- 内存不足：减少 `--batch-size`、图库规模，关闭多余进程，明确指定 CPU。
- 索引更新后 UI 未变化：服务持有内存快照，需要重启。
- 包版本解析不同：记录新的依赖与权重指纹，重新运行测试和实验，而不是沿用旧延迟报告。

容器结构参考 [FastAPI 官方部署文档](https://fastapi.tiangolo.com/deployment/docker/)。
