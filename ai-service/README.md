# ai-service

`ai-service` 是 `id-editor-tool` 中负责证件照 AI 处理的独立微服务。它保留了给未来 `id-editor-server` 使用的“路径传参接口”，同时补充了本地调试更顺手的“文件上传接口”，让你可以在本地直接上传一张照片、生成结果，并通过浏览器访问输出文件。

---

## 1. 项目定位

这个服务只负责图像处理，不负责用户、订单、支付、任务编排等业务逻辑。

当前主要能力：

- 人脸检测与基础可用性判断
- 证件照尺寸生成
- 抠图失败时的降级处理
- 背景替换或原图回退
- 结合人脸框的裁剪增强
- 轻量美化增强
- 预览图、高清图、排版图生成
- `uploads` 静态目录直出

---

## 2. 哪些接口保留给 server，哪些接口用于本地调试

### 2.1 保留给未来 `id-editor-server` 的原始接口

这些接口**继续保留**，调用方式还是 JSON + 文件路径，兼容后续 server 走共享 `uploads` 目录的模式：

- `GET /ai/health`
- `POST /ai/detect`
- `POST /ai/generate-id-photo`
- `POST /ai/generate-print-layout`

这类接口适合：

- server 已经把原图保存到共享目录
- server 只把 `uploads/original/xxx.jpg` 这类路径传给 ai-service
- ai-service 返回 `previewPath / previewUrl / hdPath / hdUrl / printPath / printUrl`

### 2.2 新增的本地调试上传接口

这些接口专门给本地快速调试使用，不替代 server 接口：

- `POST /ai/detect-upload`
- `POST /ai/generate-id-photo-upload`
- `POST /ai/generate-print-layout-upload`

这类接口适合：

- 你手头只有一张本地图片
- 想直接用 `curl` 或 Python 脚本上传
- 想立刻在浏览器里看生成结果

---

## 3. 统一响应格式

成功响应：

```json
{
  "success": true,
  "message": "OK",
  "errorCode": null,
  "data": {}
}
```

失败响应：

```json
{
  "success": false,
  "message": "Error message",
  "errorCode": "PROCESS_FAILED",
  "data": null
}
```

常见错误码：

- `INVALID_IMAGE`
- `NO_FACE_DETECTED`
- `MULTIPLE_FACES_DETECTED`
- `IMAGE_TOO_SMALL`
- `FILE_NOT_FOUND`
- `INVALID_ARGUMENT`
- `PROCESS_FAILED`

---

## 4. 环境要求

推荐环境：

- Python 3.10
- pip 23+
- Linux / macOS / WSL
- 可选：Docker 24+

Python 依赖见 `requirements.txt`，核心包括：

- FastAPI
- uvicorn
- Pillow
- OpenCV
- NumPy
- rembg
- onnxruntime
- requests

---

## 5. 配置项说明

复制 `.env.example` 为 `.env` 后即可运行。

至少支持以下配置：

| 变量名 | 说明 | 默认值 |
|---|---|---|
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务端口 | `8000` |
| `UPLOAD_ROOT` | 上传与输出根目录 | `uploads` |
| `ORIGINAL_DIR` | 原图目录 | `original` |
| `PREVIEW_DIR` | 预览图目录 | `preview` |
| `HD_DIR` | 高清图目录 | `hd` |
| `PRINT_DIR` | 排版图目录 | `print` |
| `TEMP_DIR` | 中间产物目录 | `temp` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `MAX_UPLOAD_MB` | 上传大小限制 | `15` |
| `DEFAULT_BG_COLOR` | 默认背景色 | `white` |
| `DEFAULT_LAYOUT_TYPE` | 默认排版类型 | `six` |
| `SAVE_INTERMEDIATE` | 是否保留中间文件 | `true` |
| `SEGMENTATION_ENABLED` | 是否启用 rembg 抠图 | `false` |

说明：

- 本地最简单的模式是直接使用默认 `uploads`。
- Docker 中也可以将宿主机目录挂载到容器内 `/app/uploads`。
- 如果没有启用 `SEGMENTATION_ENABLED=true`，服务仍可运行，只是会回退到原图继续裁剪流程。

---

## 6. 输出目录说明

默认输出根目录为 `uploads/`：

- `uploads/original/`：上传原图
- `uploads/preview/`：预览图
- `uploads/hd/`：高清图
- `uploads/print/`：排版图
- `uploads/temp/`：中间产物

常见命名示例：

- `uploads/original/demo_abcd1234.jpg`
- `uploads/preview/demo_preview.jpg`
- `uploads/hd/demo_hd.jpg`
- `uploads/print/demo_print_6.jpg`
- `uploads/temp/demo_segmented.png`

---

## 7. 静态访问说明

服务启动后会把 `uploads` 挂载为静态目录。

因此以下地址可直接访问：

- `http://127.0.0.1:8000/uploads/original/xxx.jpg`
- `http://127.0.0.1:8000/uploads/preview/xxx.jpg`
- `http://127.0.0.1:8000/uploads/hd/xxx.jpg`
- `http://127.0.0.1:8000/uploads/print/xxx.jpg`

接口返回里会同时给出：

- `previewPath`：给 server 用的相对存储路径，例如 `uploads/preview/xxx.jpg`
- `previewUrl`：本地浏览器直接访问的 URL 路径，例如 `/uploads/preview/xxx.jpg`
- `hdPath`
- `hdUrl`
- `printPath`
- `printUrl`

如果你在浏览器里查看，把 `previewUrl` 前面加上 `http://127.0.0.1:8000` 即可。

例如：

```text
http://127.0.0.1:8000/uploads/preview/xxx.jpg
```

---

## 8. 本地启动步骤

```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

启动后可访问：

- Swagger：`http://127.0.0.1:8000/docs`
- OpenAPI：`http://127.0.0.1:8000/openapi.json`
- 健康检查：`http://127.0.0.1:8000/ai/health`

---

## 9. Docker 启动步骤

### 9.1 构建镜像

```bash
docker build -t id-editor-tool-ai ./ai-service
```

### 9.2 运行容器

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  --env-file ./ai-service/.env.example \
  id-editor-tool-ai
```

说明：

- 容器工作目录是 `/app`
- 默认 `UPLOAD_ROOT=uploads`，对应容器内 `/app/uploads`
- 挂载后，容器里生成的图会同步到宿主机 `./uploads`

如果你想显式指定环境变量：

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e UPLOAD_ROOT=uploads \
  -e SAVE_INTERMEDIATE=true \
  id-editor-tool-ai
```

---

## 10. 原始路径接口说明（给 server 调用）

### 10.1 `GET /ai/health`

健康检查接口。

#### curl

```bash
curl http://127.0.0.1:8000/ai/health
```

### 10.2 `POST /ai/detect`

传入已存在于共享目录中的图片路径，检测人脸与质量信息。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/detect \
  -H "Content-Type: application/json" \
  -d "{\"imageId\":\"test001\",\"imagePath\":\"uploads/original/test.jpg\"}"
```

返回中会尽量包含：

- `faceDetected`
- `faceCount`
- `faceBoxes`
- `primaryFaceBox`
- `imageWidth`
- `imageHeight`
- `qualityStatus`
- `qualityMessage`
- `suggestion`

### 10.3 `POST /ai/generate-id-photo`

根据共享目录里的原图生成证件照。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-id-photo \
  -H "Content-Type: application/json" \
  -d "{\
    \"imageId\":\"test001\",\
    \"sourceType\":\"scene\",\
    \"sceneKey\":\"passport\",\
    \"backgroundColor\":\"white\",\
    \"beautyEnabled\":false,\
    \"printLayoutType\":\"six\",\
    \"originalImagePath\":\"uploads/original/test.jpg\"\
  }"
```

### 10.4 `POST /ai/generate-print-layout`

基于已生成的高清图继续生成排版图。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-print-layout \
  -H "Content-Type: application/json" \
  -d "{\
    \"imageId\":\"test001\",\
    \"hdImagePath\":\"uploads/hd/test001_hd.jpg\",\
    \"layoutType\":\"six\"\
  }"
```

---

## 11. 上传调试接口说明（给本地调试）

### 11.1 `POST /ai/detect-upload`

上传原图后自动保存到 `uploads/original/`，再复用现有检测链路。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/detect-upload \
  -F "file=@./test.jpg"
```

### 11.2 `POST /ai/generate-id-photo-upload`

上传原图并直接生成证件照。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-id-photo-upload \
  -F "file=@./test.jpg" \
  -F "backgroundColor=blue" \
  -F "sizeName=one_inch"
```

你也可以附带：

- `imageId`
- `sceneKey`
- `sourceType`
- `customWidthMm`
- `customHeightMm`
- `beautyEnabled`
- `printLayoutType`

### 11.3 `POST /ai/generate-print-layout-upload`

上传原图后先生成高清图，再生成排版图。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-print-layout-upload \
  -F "file=@./test.jpg"
```

也支持：

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-print-layout-upload \
  -F "file=@./test.jpg" \
  -F "layoutType=six" \
  -F "backgroundColor=white" \
  -F "sizeName=one_inch"
```

---

## 12. Python 测试脚本示例

`scripts/` 目录提供了三个本地脚本：

- `scripts/test_detect_upload.py`
- `scripts/test_generate_upload.py`
- `scripts/test_print_upload.py`

### 12.1 检测上传脚本

```bash
python scripts/test_detect_upload.py ./test.jpg
```

### 12.2 生成证件照脚本

```bash
python scripts/test_generate_upload.py ./test.jpg --background-color blue --size-name one_inch
```

### 12.3 生成排版图脚本

```bash
python scripts/test_print_upload.py ./test.jpg --layout-type six --size-name one_inch
```

脚本会直接打印 JSON，成功时可从返回数据中拿到：

- `previewUrl`
- `hdUrl`
- `printUrl`
- `previewPath`
- `hdPath`
- `printPath`

---

## 13. 如何准备测试图片

建议测试图片满足：

- 单人正脸
- 头顶完整可见
- 不是多人合照
- 分辨率尽量不低于 600x800
- 光线均匀
- 尽量避免遮挡、口罩、墨镜

不建议：

- 过度裁切的自拍
- 远景全身照
- 微信压缩后的超小图
- 多人合照裁出来的头像

---

## 14. 处理结果在哪里查看

方式一：看接口 JSON 返回。

例如：

- `previewPath: uploads/preview/xxx.jpg`
- `previewUrl: /uploads/preview/xxx.jpg`

方式二：浏览器直接打开。

例如：

```text
http://127.0.0.1:8000/uploads/preview/xxx.jpg
```

方式三：去本地目录查看：

```text
uploads/original/
uploads/preview/
uploads/hd/
uploads/print/
uploads/temp/
```

---

## 15. 常见报错排查

### 15.1 `FILE_NOT_FOUND`

原因：路径接口传入的 `imagePath` 或 `originalImagePath` 对应文件不存在。

排查：

- 确认文件已经在 `uploads/original/` 里
- 确认传入的是 `uploads/original/test.jpg` 这类路径
- 确认 Docker 挂载目录正确

### 15.2 `INVALID_IMAGE`

原因：上传的不是有效图片，或文件损坏。

排查：

- 尝试重新保存为 JPG/PNG
- 避免把 PDF、HEIC、截图碎片直接传入

### 15.3 `NO_FACE_DETECTED`

原因：未识别出可用人脸。

排查：

- 换单人正脸照
- 避免远景合照
- 保证五官清晰可见

### 15.4 `MULTIPLE_FACES_DETECTED`

原因：检测到多个可用人脸框。

排查：

- 不要使用多人合照
- 不要上传背后有人像海报的图片

### 15.5 抠图相关失败但服务仍返回结果

如果 `SEGMENTATION_ENABLED=false` 或 rembg 初始化失败，服务会记录降级说明，并继续使用原图执行后续裁剪/增强流程。

这属于预期降级，不会导致整个服务不可用。

---

## 16. 当前实现的增量增强说明

这次改造不是重写，而是在现有链路上做补齐：

- 保留路径接口，不破坏未来 server 调用方式
- 新增上传接口，仅用于本地调试
- 检测结果增加 `faceDetected / faceCount / faceBoxes / primaryFaceBox / suggestion`
- 质量结果增加 `qualityStatus / qualityMessage`
- 裁剪优先使用人脸框，没有人脸框时回退中心裁剪
- 输出同时返回路径和 URL
- 增加统一异常处理和统一响应结构
- `uploads` 支持直接静态访问
- Docker 可直接构建运行

---

## 17. 一次完整本地验证流程

### 第一步：启动服务

```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 第二步：上传并生成证件照

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-id-photo-upload \
  -F "file=@./test.jpg" \
  -F "backgroundColor=blue" \
  -F "sizeName=one_inch"
```

### 第三步：打开浏览器查看结果

把响应中的 `previewUrl` 拼到本地地址，例如：

```text
http://127.0.0.1:8000/uploads/preview/your_file.jpg
```

---

## 18. 已知限制与后续建议

当前限制：

- 默认不强制开启 rembg，因此换底可能回退到原图继续流程
- 人脸检测基于 OpenCV Haar，复杂姿态和复杂背景下稳定性有限
- 质量判断目前是规则增强版，不是完整模型打分

后续建议：

- 后续可在 `segment_service` 中切换更稳定的人像分割模型
- 后续可在 `detect_service` 中接入更准确的人脸检测器
- 后续可把 `previewUrl` 扩展为带主机名的绝对 URL，由 server 或网关统一拼接

