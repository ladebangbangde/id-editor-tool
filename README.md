# id-editor-tool
Python-based AI microservice for automated ID photo generation, providing portrait detection, background removal, smart cropping, and print layout generation.


## Docker 部署

可以直接使用仓库根目录下的 `Dockerfile` 构建并运行 AI 服务容器：

```bash
docker build -t ai-id-photo-service .
docker run --rm -p 8000:8000 \
  -e APP_PORT=8000 \
  -v $(pwd)/ai-service/uploads:/app/uploads \
  ai-id-photo-service
```

说明：

- 镜像默认以 `/app` 为工作目录，并直接运行 `uvicorn main:app`。
- `uploads` 建议通过 volume 挂载到宿主机或目标容器持久化目录。
- 如果目标环境已有配置中心，也可以通过 `APP_NAME`、`APP_HOST`、`APP_PORT`、`UPLOAD_BASE_DIR` 等环境变量覆盖默认值。
