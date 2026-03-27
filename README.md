# id-editor-tool

一个可本地运行、可 Docker 启动、可直接通过 HTTP 调用的证件照处理微服务。它专注于 **照片检测、抠图换底、标准证件照生成、6 寸排版图生成、结果静态访问**，方便后续由 `id-editor-server` 通过 HTTP 接入。

## 功能列表

- `GET /health`：健康检查。
- `POST /detect`：支持上传图片或传入共享图片路径做 CPU 友好的预检（PASS/WARNING/FAIL）。
- `POST /generate`：支持上传原图或传入共享路径生成证件照预览图与高清图。
- `POST /photo/precheck`：独立的照片预检接口，返回结构化指标与风险提示。
- `POST /photo/process`：`/generate` 的兼容别名，处理前自动执行预检。
- `GET /photo/specs`：返回 tool 当前支持的 canonical `sizeKey`、像素/毫米尺寸、aliases 与是否支持自定义尺寸。
- `POST /layout`：基于原图/共享路径/已生成证件照生成 6 寸排版图。
- `POST /formal-wear`：兼容占位接口（换装功能已下线，固定返回下线提示，不影响主链路）。
- `/uploads/...`：直接访问共享上传目录中的静态结果。
- 自动按日期 + 任务 ID 保存输出，便于肉眼查看效果。
- 支持基础调试中间文件保存：前景图、透明裁剪图。
- 自带 Swagger / OpenAPI：`/docs`。

## 第一阶段实现策略

第一阶段优先保证“**可运行、可测试、可看到效果**”的主链路打通：

- 人脸检测：`MediaPipe Face Detection`（CPU，无需 GPU）。
- 抠图：默认使用百度 AI 人像分割（`foreground` 透明前景图）；仅在 `BAIDU_SEGMENTATION_ENABLED=false` 时才走 rembg 调试链路。
- 换底：Pillow 合成白/蓝/红底。
- 标准裁剪：基于人脸框和尺寸比例的规则裁剪。
- 增强：轻量亮度/对比度/锐化处理。
- 排版：6 寸纸张打印参考图。
- 预检指标：`OpenCV` 计算清晰度、亮度、边缘密度等可解释指标。
- 质量提示：提供 `primaryIssue/primaryMessage/secondaryWarnings/qualityStatus`，并支持一阶段颈部饰品检测。
- 普通图与高清图分离：preview 使用更低分辨率 JPEG 压缩，hd 保留完整输出像素。

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
  uploads/
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

如需走共享目录联调，请把测试照片放到共享上传根目录下，例如：

```bash
mkdir -p uploads/original
cp /path/to/your-photo.jpg uploads/original/test.jpg
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

该命令会利用 `skimage` 自带的 `astronaut` 示例图生成测试图片；如需和 Docker 共享目录联调，建议再复制到 `uploads/original/`。

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
  -v $(pwd)/uploads:/app/uploads \
  id-editor-tool
```

> 如果与 `id-editor-server` 联动，请对两个容器都挂载宿主机上的同一个目录到 `/app/uploads`。
启动后访问：

- <http://127.0.0.1:8000/health>
- <http://127.0.0.1:8000/docs>

### 与 `id-editor-server` 共享宿主机目录

推荐让两个容器都挂载同一个宿主机目录：

```bash
mkdir -p /data/id-photo-uploads/{original,preview,hd,print,temp}

docker run -d --name id-editor-tool \
  -p 8000:8000 \
  -v /data/id-photo-uploads:/app/uploads \
  id-editor-tool
```

此时 `id-editor-server` 可以把 `/app/uploads/original/xxx.png` 直接传给本服务的 `imagePath` / `idPhotoPath` 字段，本服务会继续把生成结果写入同一挂载目录并通过 `/uploads/...` 暴露。

## 配置说明

通过 `.env` 配置：

```env
SERVICE_NAME=id-editor-tool
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
DEBUG=false
UPLOAD_ROOT=/app/uploads
STATIC_MOUNT_PATH=/uploads
ORIGINAL_DIR=original
PREVIEW_DIR=preview
HD_DIR=hd
PRINT_DIR=print
TEMP_DIR=temp
SAVE_INTERMEDIATE=true
BAIDU_SEGMENTATION_ENABLED=true
BAIDU_API_KEY=
BAIDU_SECRET_KEY=
BAIDU_OAUTH_URL=https://aip.baidubce.com/oauth/2.0/token
BAIDU_SEGMENTATION_URL=https://aip.baidubce.com/rest/2.0/image-classify/v1/body_seg
BAIDU_HTTP_TIMEOUT_SEC=15
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
- `UPLOAD_ROOT`：容器内共享上传根目录，默认 `/app/uploads`
- `STATIC_MOUNT_PATH`：静态访问前缀，默认 `/uploads`
- `SAVE_INTERMEDIATE`：是否保存抠图等中间结果（包含 `baidu_foreground.png` / `baidu_labelmap.png` / `baidu_scoremap.png`）
- `BAIDU_SEGMENTATION_ENABLED`：是否启用百度人像分割正式链路（默认 true）
- `BAIDU_API_KEY` / `BAIDU_SECRET_KEY`：百度鉴权凭据；缺失时会直接报错，不会静默回退
- `MAX_UPLOAD_SIZE_MB`：最大上传大小
- `DEFAULT_BACKGROUND_COLOR`：默认底色
- `DEFAULT_SIZE_KEY`：默认证件照规格

## 证件照规格

当前内置：

- `one_inch`：一寸，`295x413`
- `small_one_inch`：小一寸，`260x378`
- `two_inch`：二寸，`413x579`
- `passport_photo`：护照，`390x567`

常见别名（自动映射）：

- `passport` → `passport_photo`
- `visa` → `passport_photo`

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
  "status": "ok",
  "uploadRoot": "/app/uploads",
  "staticMountPath": "/uploads",
  "directories": {
    "original": "/app/uploads/original",
    "preview": "/app/uploads/preview",
    "hd": "/app/uploads/hd",
    "print": "/app/uploads/print",
    "temp": "/app/uploads/temp"
  }
}
```

### 2. 图片检测

`POST /detect`

- Content-Type: `multipart/form-data`
- 字段：`file` 或 `imagePath`
- `imagePath` 可直接传共享绝对路径，如 `/app/uploads/original/test.jpg`

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

- `file`：原图，可选
- `imagePath`：共享目录中的原图路径，可选，例如 `/app/uploads/original/test.jpg`
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
    "previewPath": "/app/uploads/preview/20260318/.../id_photo_preview.jpg",
    "previewUrl": "/uploads/preview/20260318/.../id_photo_preview.jpg",
    "hdPath": "/app/uploads/hd/20260318/.../id_photo_hd.png",
    "hdUrl": "/uploads/hd/20260318/.../id_photo_hd.png",
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

1. 直接传 `idPhoto` 或 `idPhotoPath`
2. 传 `image` / `imagePath` + `sizeKey` 等参数，让服务内部先生成证件照再排版

表单字段：

- `idPhoto`：已生成证件照上传文件，可选
- `idPhotoPath`：共享目录中的已生成证件照绝对路径，可选
- `image`：原图上传文件，可选
- `imagePath`：共享目录中的原图绝对路径，可选
- `sceneId` 或 `sizeKey`
- `backgroundColor`
- `enhance`
- `saveOutput`
- `paper`，当前支持 `6inch`

### 5. 历史换装接口兼容占位

`POST /formal-wear`

表单字段：

- `file`：原图上传文件，可选
- `imagePath`：共享目录中的原图绝对路径，可选
- `gender`
- `style`
- `color`
- `enhance`
- `saveOutput`

说明：

- 为避免 server 误调用旧接口导致 500，当前保留 `POST /formal-wear` 占位路由。
- 该接口固定返回 `FORMAL_WEAR_OFFLINE` 与“换装功能已下线”提示，不再执行任何换装推理和任务处理。
- `/detect`、`/generate`、`/layout` 证件照主链路不受影响。

示例返回重点字段：

```json
{
  "success": false,
  "message": "换装功能已下线",
  "error": {
    "code": "FORMAL_WEAR_OFFLINE",
    "message": "换装功能已下线"
  },
  "data": {
    "status": "offline",
    "message": "换装功能已下线"
  }
}
```

### 6. 静态输出访问

生成完成后，可直接访问：

```text
http://127.0.0.1:8000/uploads/preview/20260318/<task_id>/id_photo_preview.jpg
http://127.0.0.1:8000/uploads/hd/20260318/<task_id>/id_photo_hd.png
http://127.0.0.1:8000/uploads/print/20260318/<task_id>/layout_6inch.jpg
```

## curl 快速验证

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### detect

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@uploads/original/test.jpg"
```

或直接传共享绝对路径：

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -F "imagePath=/app/uploads/original/test.jpg"
```

### generate

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "file=@uploads/original/test.jpg" \
  -F "sizeKey=one_inch" \
  -F "backgroundColor=blue" \
  -F "enhance=true" \
  -F "saveOutput=true"
```

或直接传共享绝对路径：

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "imagePath=/app/uploads/original/test.jpg" \
  -F "sizeKey=one_inch" \
  -F "backgroundColor=blue" \
  -F "saveOutput=true"
```

### layout（直接用原图）

```bash
curl -X POST http://127.0.0.1:8000/layout \
  -F "image=@uploads/original/test.jpg" \
  -F "sizeKey=one_inch" \
  -F "backgroundColor=blue" \
  -F "paper=6inch" \
  -F "saveOutput=true"
```

或使用共享绝对路径：

```bash
curl -X POST http://127.0.0.1:8000/layout \
  -F "imagePath=/app/uploads/original/test.jpg" \
  -F "sizeKey=one_inch" \
  -F "backgroundColor=blue" \
  -F "paper=6inch" \
  -F "saveOutput=true"
```

### formal-wear（兼容占位）

```bash
curl -X POST http://127.0.0.1:8000/formal-wear \
  -F "imagePath=/app/uploads/original/test.jpg" \
  -F "gender=female" \
  -F "style=formal" \
  -F "color=blue" \
  -F "enhance=false" \
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

所有结果都会保存在共享上传根目录 `/app/uploads` 下：

```text
/app/uploads/original/
/app/uploads/preview/<日期>/<task_id>/id_photo_preview.jpg
/app/uploads/hd/<日期>/<task_id>/id_photo_hd.png
/app/uploads/print/<日期>/<task_id>/layout_6inch.jpg
/app/uploads/temp/<日期>/<task_id>/foreground.png
```

你可以：

1. 直接在宿主机挂载目录查看文件
2. 或访问返回里的 `/uploads/...` URL 在浏览器查看

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

- 系统会先对候选人脸框做过滤、去重与主脸优先选择，再统计有效独立人脸数量
- 只有过滤后仍存在 2 张及以上有效独立人脸时，才会返回 `MULTIPLE_FACES_DETECTED`
- 如是单人图，建议仍尽量避免帽檐、头发遮挡、耳机、强阴影或复杂高对比背景
- 请裁掉其他人，只保留单人照片

### 3. 返回 `IMAGE_TOO_SMALL`

- 请换更高分辨率图片
- 当前默认至少 `400x400`

### 4. 首次运行抠图较慢

- `rembg` / `onnxruntime` 首次初始化通常比后续慢
- Docker 镜像构建阶段会预置 `u2net.onnx`，运行时默认不再联网下载模型

### 5. Docker 中看不到输出图

确认运行时挂载了卷：

```bash
-v $(pwd)/uploads:/app/uploads
```

## 最小验收流程

推荐按下面顺序验证：

1. 启动服务
2. `GET /health`
3. `POST /detect`
4. `POST /generate`
5. 打开返回的 `/uploads/...` 查看证件照
6. `POST /layout`
7. 打开返回的排版图 URL

这样即可完成“**本地可运行、可测试、可看到效果**”的第一阶段验收。
