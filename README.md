# id-editor-tool

一个可本地运行、可 Docker 启动、可直接通过 HTTP 调用的证件照处理微服务。它专注于 **照片检测、抠图换底、标准证件照生成、6 寸排版图生成、结果静态访问**，方便后续由 `id-editor-server` 通过 HTTP 接入。

## 功能列表

- `GET /health`：健康检查。
- `POST /detect`：上传图片做人脸基础检测。
- `POST /generate`：上传原图生成证件照预览图与高清图。
- `POST /layout`：基于原图或已生成证件照生成 6 寸排版图。
- `/outputs/...`：直接访问本地输出结果。
- 自动按日期 + 任务 ID 保存输出，便于肉眼查看效果。
- 支持基础调试中间文件保存：前景图、透明裁剪图。
- 自带 Swagger / OpenAPI：`/docs`。

## 第一阶段实现策略

第一阶段优先保证“**可运行、可测试、可看到效果**”的主链路打通：

- 人脸检测：`scikit-image` 内置 Haar Cascade（离线运行）。
- 抠图：`rembg` + `onnxruntime` 本地离线推理。
- 换底：Pillow 合成白/蓝/红底。
- 标准裁剪：基于人脸框和尺寸比例的规则裁剪。
- 增强：轻量亮度/对比度/锐化处理。
- 排版：6 寸纸张打印参考图。

## 目录结构

```text
id-editor-tool/
  app/
    api/
    core/
    schemas/
    services/
    utils/
    main.py
  inputs/
  outputs/
  scripts/
    run_local.sh
    test_detect.py
    test_generate.py
    test_layout.py
    create_sample_image.py
  ai-service/               # 保留的旧原型目录，不作为本阶段主入口
  requirements.txt
  Dockerfile
  .env.example
  README.md
```

## 环境要求

- Python 3.11 推荐（3.10+ 通常也可运行）
- Linux / macOS / WSL 推荐
- 建议至少 4GB 内存（首次使用 `rembg` / onnxruntime 时更稳）

## 准备测试图片

请把你的测试照片放到 `inputs/` 目录，例如：

```bash
cp /path/to/your-photo.jpg inputs/test.jpg
```

建议照片要求：

- 单人正面半身或头像
- 光线尽量均匀
- 分辨率至少 `400x400`
- 避免多人合照、严重遮挡、过度模糊

> 仓库没有内置真人测试图，避免提交真实人像。你只需要自行放入一张照片即可完成全部流程验证。

如果你只是想快速试跑，也可以生成一个示例头像：

```bash
python scripts/create_sample_image.py
```

该命令会利用 `skimage` 自带的 `astronaut` 示例图生成 `inputs/sample_astronaut.png`。

## 本地启动

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. 启动服务

```bash
./scripts/run_local.sh
```

或：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 打开文档

- Swagger UI: <http://127.0.0.1:8000/docs>
- 健康检查: <http://127.0.0.1:8000/health>

## Docker 启动

### 构建镜像

```bash
docker build -t id-editor-tool .
```

### 运行容器

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/inputs:/app/inputs \
  id-editor-tool
```

启动后访问：

- <http://127.0.0.1:8000/health>
- <http://127.0.0.1:8000/docs>

## 配置说明

通过 `.env` 配置：

```env
SERVICE_NAME=id-editor-tool
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
DEBUG=false
INPUT_DIR=inputs
OUTPUT_DIR=outputs
SAVE_INTERMEDIATE=true
MAX_UPLOAD_SIZE_MB=15
DEFAULT_BACKGROUND_COLOR=blue
DEFAULT_SIZE_KEY=one_inch
DEFAULT_LAYOUT_PAPER=6inch
MIN_IMAGE_WIDTH=400
MIN_IMAGE_HEIGHT=400
PREVIEW_QUALITY=90
HD_QUALITY=95
```

关键配置项：

- `APP_PORT`：服务端口
- `INPUT_DIR`：输入目录
- `OUTPUT_DIR`：输出目录
- `SAVE_INTERMEDIATE`：是否保存抠图等中间结果
- `MAX_UPLOAD_SIZE_MB`：最大上传大小
- `DEFAULT_BACKGROUND_COLOR`：默认底色
- `DEFAULT_SIZE_KEY`：默认证件照规格

## 证件照规格

当前内置：

- `one_inch`：一寸，`295x413`
- `small_one_inch`：小一寸，`260x378`
- `two_inch`：二寸，`413x579`

当前支持底色：

- `white`
- `blue`
- `red`

## API 说明

### 1. 健康检查

`GET /health`

示例返回：

```json
{
  "success": true,
  "service": "id-editor-tool",
  "status": "ok"
}
```

### 2. 图片检测

`POST /detect`

- Content-Type: `multipart/form-data`
- 字段：`file`

示例返回：

```json
{
  "success": true,
  "message": "ok",
  "data": {
    "hasFace": true,
    "faceCount": 1,
    "width": 1200,
    "height": 1600,
    "pass": true,
    "reasons": [],
    "faceBoxes": [{"x": 320, "y": 180, "width": 420, "height": 420}],
    "recommended": true,
    "warning": null
  }
}
```

失败时统一返回：

```json
{
  "success": false,
  "message": "error",
  "error": {
    "code": "NO_FACE_DETECTED",
    "message": "No face detected in the uploaded image"
  }
}
```

### 3. 证件照生成

`POST /generate`

表单字段：

- `file`：原图
- `sceneId` 或 `sizeKey`
- `backgroundColor`
- `enhance`
- `saveOutput`

示例返回重点字段：

```json
{
  "success": true,
  "message": "ok",
  "data": {
    "taskId": "gen_20260318_120000_ab12cd34",
    "previewPath": "/workspace/id-editor-tool/outputs/20260318/.../id_photo_preview.jpg",
    "previewUrl": "/outputs/20260318/.../id_photo_preview.jpg",
    "hdPath": "/workspace/id-editor-tool/outputs/20260318/.../id_photo_hd.png",
    "hdUrl": "/outputs/20260318/.../id_photo_hd.png",
    "backgroundColor": "blue",
    "size": {
      "key": "one_inch",
      "name": "一寸",
      "widthPx": 295,
      "heightPx": 413,
      "widthMm": 25.0,
      "heightMm": 35.0
    },
    "warnings": []
  }
}
```

### 4. 排版图生成

`POST /layout`

两种调用方式：

1. 直接传 `idPhoto`
2. 传 `image` + `sizeKey` 等参数，让服务内部先生成证件照再排版

表单字段：

- `idPhoto`：已生成证件照，可选
- `image`：原图，可选
- `sceneId` 或 `sizeKey`
- `backgroundColor`
- `enhance`
- `saveOutput`
- `paper`，当前支持 `6inch`

### 5. 静态输出访问

生成完成后，可直接访问：

```text
http://127.0.0.1:8000/outputs/20260318/<task_id>/id_photo_preview.jpg
http://127.0.0.1:8000/outputs/20260318/<task_id>/id_photo_hd.png
http://127.0.0.1:8000/outputs/20260318/<task_id>/layout_6inch.jpg
```

## curl 快速验证

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### detect

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@inputs/test.jpg"
```

### generate

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "file=@inputs/test.jpg" \
  -F "sizeKey=one_inch" \
  -F "backgroundColor=blue" \
  -F "enhance=true" \
  -F "saveOutput=true"
```

### layout（直接用原图）

```bash
curl -X POST http://127.0.0.1:8000/layout \
  -F "image=@inputs/test.jpg" \
  -F "sizeKey=one_inch" \
  -F "backgroundColor=blue" \
  -F "paper=6inch" \
  -F "saveOutput=true"
```

## Python 测试脚本

### detect

```bash
python scripts/test_detect.py --image inputs/test.jpg
```

### generate

```bash
python scripts/test_generate.py --image inputs/test.jpg --size-key one_inch --background-color blue --enhance
```

### layout

```bash
python scripts/test_layout.py --image inputs/test.jpg --size-key one_inch --background-color blue
```

## 输出结果在哪里看

所有结果保存在：

```text
outputs/<日期>/<task_id>/
```

典型文件：

- `foreground.png`：抠图前景（启用 `SAVE_INTERMEDIATE=true` 时保存）
- `cropped_rgba.png`：透明裁剪图（启用 `SAVE_INTERMEDIATE=true` 时保存）
- `id_photo_preview.jpg`：预览图
- `id_photo_hd.png`：高清图
- `layout_6inch.jpg`：排版图

你可以：

1. 直接打开本地文件夹查看
2. 或访问返回里的 `/outputs/...` URL 在浏览器查看

## 日志与调试

服务会记录关键步骤：

- 收到上传
- 完成人脸检测
- 完成抠图
- 完成换底
- 完成裁剪
- 完成排版
- 输出文件路径

如果需要保留更多处理痕迹，请在 `.env` 中设置：

```env
SAVE_INTERMEDIATE=true
```

## 错误码

当前统一业务错误码：

- `INVALID_IMAGE`
- `NO_FACE_DETECTED`
- `MULTIPLE_FACES_DETECTED`
- `IMAGE_TOO_SMALL`
- `PROCESS_FAILED`
- `INVALID_ARGUMENT`

## 常见问题与排查

### 1. 返回 `NO_FACE_DETECTED`

- 尝试换一张单人正面照片
- 避免强逆光、遮挡、过暗
- 建议头像占画面更大一些

### 2. 返回 `MULTIPLE_FACES_DETECTED`

- 请裁掉其他人，只保留单人照片

### 3. 返回 `IMAGE_TOO_SMALL`

- 请换更高分辨率图片
- 当前默认至少 `400x400`

### 4. 首次运行抠图较慢

- `rembg` / `onnxruntime` 首次初始化通常比后续慢
- Docker 或低配机器首次耗时更明显

### 5. Docker 中看不到输出图

确认运行时挂载了卷：

```bash
-v $(pwd)/outputs:/app/outputs
```

## 最小验收流程

推荐按下面顺序验证：

1. 启动服务
2. `GET /health`
3. `POST /detect`
4. `POST /generate`
5. 打开返回的 `/outputs/...` 查看证件照
6. `POST /layout`
7. 打开返回的排版图 URL

这样即可完成“**本地可运行、可测试、可看到效果**”的第一阶段验收。
