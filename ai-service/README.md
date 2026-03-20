# ai-service

`ai-service` 是 `id-editor-tool` 中负责证件照 AI 处理的独立微服务。当前阶段保留给未来 `id-editor-server` 使用的“路径传参接口”，同时补充本地联调更顺手的“upload-debug 文件上传接口”，方便你直接上传图片、调试返回结构，并通过浏览器查看生成结果。

---

## 1. 当前能力范围

当前已经接入并保留的正式处理链路：

- `GET /ai/health`
- `GET /ai/colors`
- `GET /ai/photo-sizes`
- `POST /ai/detect`
- `POST /ai/generate-id-photo`
- `POST /ai/generate-print-layout`

当前 `generate-id-photo` 主链继续复用既有处理步骤，不做重写：

- detect
- segment
- background
- crop
- enhance
- quality
- optional print

当前背景色能力：

- `white`
- `blue`
- `red`

当前内置尺寸模板 / `sizeName` / `sceneKey` 支持：

- `one_inch`
- `two_inch`
- `small_two_inch`
- `passport`
- `visa`
- `driver_license`

当前排版类型 / `layoutType` 支持：

- `six`
- `eight`
- `twelve`

---

## 2. 正式接口 vs 本地调试接口

### 2.1 正式接口：给 server 调用

这些接口给未来 `id-editor-server` 或其它后端服务调用，使用 JSON + 已保存文件路径的方式，不改变现有处理链路：

- `GET /ai/health`
- `GET /ai/colors`
- `GET /ai/photo-sizes`
- `POST /ai/detect`
- `POST /ai/generate-id-photo`
- `POST /ai/generate-print-layout`

特点：

- server 先把原图保存到共享目录，例如 `uploads/original/test.jpg`
- server 再把 `originalImagePath` 或 `hdImagePath` 传给 ai-service
- ai-service 返回 `previewPath / previewUrl / hdPath / hdUrl / printPath / printUrl`

### 2.2 upload-debug 接口：仅给本地调试

以下接口是**本地调试接口**，便于你在开发机上直接上传图片验证，不替代正式接口：

- `POST /ai/detect-upload`
- `POST /ai/generate-id-photo-upload`
- `POST /ai/generate-print-layout-upload`

特点：

- 调试时直接上传文件即可
- 服务会先保存原图到 `uploads/original/`
- 然后复用正式 detect / generate / print pipeline
- 不影响未来 server 走路径传参的正式调用方式

Swagger 中可直接看到这两类接口：

- `config`
- `detect`
- `generate`
- `print`
- `upload-debug`

---

## 3. 日志分级与开发者可观测性

当前 ai-service 已补充一套面向开发者的日志输出体系：

- `INFO`：记录请求开始/结束、上传保存、detect/generate/print 各关键阶段完成、输出文件落盘
- `DEBUG`：记录路径解析、OpenCV 检测细节、临时文件清理等更细节的调试信息
- `WARNING`：记录校验失败、抠图回退、配置关闭、文件过大、模板不合法等可恢复问题
- `ERROR` / `EXCEPTION`：记录 5xx 级异常、依赖导入失败、未处理异常

日志中会尽量带上这些上下文，方便开发者排查一次请求完整链路：

- `request_id`
- `image_id`
- `path` / `method`
- `background_color`
- `scene_key` / `layout_type`
- `face_count` / `validation_passed`
- `segmentation_enabled` / `fallback_used`
- `duration_ms`

如果你希望本地看到更细的阶段日志，建议设置：

```env
LOG_LEVEL=DEBUG
```

---

## 4. 统一响应格式

所有接口统一返回：

```json
{
  "success": true,
  "message": "OK",
  "errorCode": null,
  "data": {}
}
```

失败时：

```json
{
  "success": false,
  "message": "Error message",
  "errorCode": "PROCESS_FAILED",
  "data": null
}
```

---

## 5. 只读配置接口

### 5.1 `GET /ai/colors`

返回当前所有可用背景色，直接复用 `constants/colors.py`。

#### curl

```bash
curl http://127.0.0.1:8000/ai/colors
```

返回 `data` 中会包含：

- `key`
- `nameZh`
- `nameEn`
- `rgb`
- `hex`
- `description`

### 5.2 `GET /ai/photo-sizes`

返回当前所有证件照尺寸模板，直接复用 `constants/photo_sizes.py`。

#### curl

```bash
curl http://127.0.0.1:8000/ai/photo-sizes
```

返回 `data` 中会包含：

- `sceneKey`
- `sceneName`
- `sceneNameEn`
- `widthMm`
- `heightMm`
- `pixelWidth`
- `pixelHeight`
- `unit`
- `description`

---

## 6. 正式接口说明（给 server 调用）

### 6.1 `GET /ai/health`

```bash
curl http://127.0.0.1:8000/ai/health
```

### 6.2 `POST /ai/detect`

传入共享目录中的图片路径，输出检测结果与可读校验信息。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/detect \
  -H "Content-Type: application/json" \
  -d '{
    "imageId":"test001",
    "originalImagePath":"uploads/original/test.jpg"
  }'
```

当前返回结构会明确包含：

- `imageId`
- `originalImagePath`
- `originalImageUrl`
- `faceDetected`
- `faceCount`
- `primaryFaceBox`
- `faceBoxes`
- `imageWidth`
- `imageHeight`
- `validationPassed`
- `reasons`
- `suggestion`

此外还会尽量补充：

- `imageFormat`
- `imageMode`
- `blurScore`
- `poseValid`
- `occlusionDetected`
- `qualityStatus`
- `qualityMessage`
- `message`

### 6.3 `POST /ai/generate-id-photo`

根据共享目录里的原图生成证件照。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-id-photo \
  -H "Content-Type: application/json" \
  -d '{
    "imageId":"test001",
    "sourceType":"scene",
    "sceneKey":"passport",
    "backgroundColor":"white",
    "beautyEnabled":false,
    "printLayoutType":"six",
    "originalImagePath":"uploads/original/test.jpg"
  }'
```

当前返回会保留原有字段，并额外明确：

- `whetherFallbackUsed`
- `segmentationSucceeded`
- `finalOutputType`
- `canDirectlyUseForRegistration`

常见字段：

- `backgroundColor`
- `method`
- `qualityStatus`
- `qualityMessage`
- `cropBox`
- `headRatio`
- `appliedOperations`
- `processNotes`
- `previewPath`
- `previewUrl`
- `hdPath`
- `hdUrl`
- `printPath`
- `printUrl`

### 6.4 `POST /ai/generate-print-layout`

基于已生成高清图继续生成排版图。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-print-layout \
  -H "Content-Type: application/json" \
  -d '{
    "imageId":"test001",
    "hdImagePath":"uploads/hd/test001_hd.jpg",
    "layoutType":"six"
  }'
```

返回结构会明确包含：

- `printPath`
- `printUrl`
- `layoutType`
- `paperType`
- `photoCount`

---

## 7. upload-debug 接口说明（仅用于本地调试）

### 7.1 `POST /ai/detect-upload`

上传原图后自动保存到 `uploads/original/`，再复用正式检测链路。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/detect-upload \
  -F "file=@./test.jpg"
```

#### Python requests 脚本

```bash
python scripts/test_detect_upload.py ./test.jpg
```

### 7.2 `POST /ai/generate-id-photo-upload`

上传原图并直接生成证件照。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-id-photo-upload \
  -F "file=@./test.jpg" \
  -F "backgroundColor=blue" \
  -F "sizeName=one_inch"
```

可选表单字段：

- `imageId`
- `sourceType`
- `sceneKey`
- `sizeName`
- `customWidthMm`
- `customHeightMm`
- `backgroundColor`
- `beautyEnabled`
- `printLayoutType`

#### Python requests 脚本

```bash
python scripts/test_generate_upload.py ./test.jpg --background-color blue --size-name one_inch
```

### 7.3 `POST /ai/generate-print-layout-upload`

上传原图后先走证件照生成，再继续生成排版图。

#### curl

```bash
curl -X POST http://127.0.0.1:8000/ai/generate-print-layout-upload \
  -F "file=@./test.jpg" \
  -F "layoutType=six" \
  -F "backgroundColor=white" \
  -F "sizeName=one_inch"
```

#### Python requests 脚本

```bash
python scripts/test_print_upload.py ./test.jpg --layout-type six --size-name one_inch
```

---

## 8. 生成结果如何通过浏览器访问

服务启动后会把 `uploads` 挂载为静态目录，因此以下路径可以在浏览器中直接打开：

- `http://127.0.0.1:8000/uploads/original/xxx.jpg`
- `http://127.0.0.1:8000/uploads/preview/xxx.jpg`
- `http://127.0.0.1:8000/uploads/hd/xxx.jpg`
- `http://127.0.0.1:8000/uploads/print/xxx.jpg`

接口返回中会同时给出：

- `originalImagePath / originalImageUrl`
- `previewPath / previewUrl`
- `hdPath / hdUrl`
- `printPath / printUrl`

其中：

- `xxxPath` 适合 server 或存储系统继续传递
- `xxxUrl` 适合本地浏览器直接访问

例如返回：

```json
{
  "previewUrl": "/uploads/preview/test001_preview.jpg"
}
```

则浏览器可访问：

```text
http://127.0.0.1:8000/uploads/preview/test001_preview.jpg
```

---

## 9. 本地启动

```bash
cd ai-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

启动后可访问：

- Swagger：`http://127.0.0.1:8000/docs`
- OpenAPI：`http://127.0.0.1:8000/openapi.json`
- 健康检查：`http://127.0.0.1:8000/ai/health`

---

## 10. 本地验证步骤

### 10.1 验证只读配置接口

```bash
curl http://127.0.0.1:8000/ai/colors
curl http://127.0.0.1:8000/ai/photo-sizes
```

### 10.2 验证正式 detect 接口

先准备一张图片到：

```text
uploads/original/test.jpg
```

然后执行：

```bash
curl -X POST http://127.0.0.1:8000/ai/detect \
  -H "Content-Type: application/json" \
  -d '{"imageId":"demo001","originalImagePath":"uploads/original/test.jpg"}'
```

### 10.3 验证 upload-debug 检测接口

```bash
python scripts/test_detect_upload.py ./test.jpg
```

### 10.4 验证 upload-debug 证件照生成接口

```bash
python scripts/test_generate_upload.py ./test.jpg --background-color red --size-name passport
```

### 10.5 验证 upload-debug 排版接口

```bash
python scripts/test_print_upload.py ./test.jpg --layout-type six --size-name one_inch
```

### 10.6 浏览器查看生成文件

把返回中的 `previewUrl`、`hdUrl`、`printUrl` 拼到本地地址即可，例如：

```text
http://127.0.0.1:8000/uploads/print/demo001_print_6.jpg
```

---

## 11. 说明

这一轮是“第二阶段增量增强”，不是重写：

- 保留原有正式接口路径
- 保留既有处理链与服务拆分
- 新增只读配置接口，便于 server / UI 拉取配置
- 增强 detect / generate / print 的对外返回结构
- 强化 upload-debug 说明，提升本地调试体验
