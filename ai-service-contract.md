# AI Service Interface Contract

## 1. 文档目标

本文档用于约束主业务后端（server）与 AI 图像处理服务（ai-service）之间的接口协议，确保两端在以下方面保持一致：

- API 路径
- 请求字段
- 返回字段
- 错误处理
- 文件输出规则
- 字段命名规范
- 结果数据映射关系

AI 服务是下游服务，由主业务后端通过 HTTP 调用。前端不得直接调用 AI 服务。

---

## 2. 调用关系

调用链路如下：

WeChat Mini Program -> server -> ai-service

说明：

- 小程序只调用主业务后端 `server`
- `server` 负责用户、订单、权限、历史记录
- `ai-service` 只负责图像处理
- `ai-service` 不负责支付、权限、下载控制

---

## 3. 基础约定

### 3.1 协议
- HTTP / JSON
- 编码：UTF-8

### 3.2 AI 服务基础地址
示例：

`http://localhost:8000`

生产环境由配置项决定，例如：

`AI_SERVICE_BASE_URL=http://127.0.0.1:8000`

### 3.3 Content-Type
统一使用：

`application/json`

---

## 4. 字段命名规范

所有字段统一使用 camelCase。

例如：

- imageId
- sourceType
- sceneKey
- customWidthMm
- customHeightMm
- backgroundColor
- beautyEnabled
- printLayoutType
- originalImagePath
- previewUrl
- hdUrl
- printUrl
- qualityStatus

不得擅自改为：

- snake_case
- kebab-case
- 其他不一致命名

---

## 5. 统一返回格式

AI 服务所有接口都必须返回以下统一结构。

### 成功
```json
{
  "success": true,
  "message": "OK",
  "data": {}
}
```

### 失败
```json
{
  "success": false,
  "message": "Error message",
  "data": null
}
```

说明：

- `success` 只表示本次接口调用是否成功
- 业务详细结果写在 `data`
- 不允许返回不带 `success` 的结构
- 不允许返回数组作为根节点
- 不允许返回字段名不统一的结果

---

## 6. 文件与路径约定

AI 服务生成的输出资源路径必须稳定，方便主业务后端写入数据库。

### 6.1 输出目录
- 预览图：`uploads/preview/`
- 高清图：`uploads/hd/`
- 排版图：`uploads/print/`
- 临时文件：`uploads/temp/`

### 6.2 文件命名规则
必须基于 `imageId` 生成，保证可追踪性。

示例：

- `uploads/preview/img_001_preview.jpg`
- `uploads/hd/img_001_hd.jpg`
- `uploads/print/img_001_print_6.jpg`
- `uploads/print/img_001_print_8.jpg`
- `uploads/print/img_001_print_12.jpg`

### 6.3 路径字段说明
AI 服务返回的路径字段可先返回相对路径，例如：

```json
{
  "previewUrl": "uploads/preview/img_001_preview.jpg"
}
```

由主业务后端决定是否转换成完整可访问 URL。

---

## 7. 场景模板约定

当 `sourceType = scene` 时，必须使用以下 `sceneKey` 之一：

- `one_inch`
- `two_inch`
- `passport`
- `visa`
- `driver_license`
- `resume`
- `exam`

### 7.1 说明
- `sceneKey` 必须与主业务后端的 `scene_templates.scene_key` 保持一致
- AI 服务内部不得随意改名
- 若 `sceneKey` 无效，应返回错误

### 7.2 自定义尺寸模式
当 `sourceType = custom` 时：
- `sceneKey` 可为空
- 必须提供：
  - `customWidthMm`
  - `customHeightMm`

---

## 8. 枚举值约定

### 8.1 sourceType
允许值：
- `scene`
- `custom`

### 8.2 backgroundColor
允许值：
- `white`
- `blue`
- `red`

### 8.3 printLayoutType
允许值：
- `six`
- `eight`
- `twelve`

### 8.4 qualityStatus
允许值：
- `passed`
- `warning`
- `failed`

---

## 9. AI 服务接口列表

AI 服务必须提供以下接口：

- `POST /ai/detect`
- `POST /ai/generate-id-photo`
- `POST /ai/generate-print-layout`
- `GET /ai/health`

---

## 10. 接口详细契约

### 10.1 健康检查

#### 请求
`GET /ai/health`

#### 返回
```json
{
  "success": true,
  "message": "AI service is running",
  "data": {
    "service": "ai-id-photo-service"
  }
}
```

#### 用途
- `server` 启动后可用于检查 AI 服务是否可用
- 可作为部署探活接口

---

### 10.2 照片检测接口

#### 请求
`POST /ai/detect`

#### 请求体
```json
{
  "imageId": "img_001",
  "originalImagePath": "uploads/original/img_001.jpg"
}
```

#### 字段说明
- `imageId`：图片唯一标识，由主业务后端生成
- `originalImagePath`：原图文件路径

#### 成功返回
```json
{
  "success": true,
  "message": "OK",
  "data": {
    "imageId": "img_001",
    "hasFace": true,
    "faceCount": 1,
    "blurScore": 0.91,
    "poseValid": true,
    "occlusionDetected": false,
    "message": "照片可用于制作证件照"
  }
}
```

#### data 字段说明
- `imageId`：请求中的 imageId 原样返回
- `hasFace`：是否检测到人脸
- `faceCount`：人脸数量
- `blurScore`：模糊评分，0~1 或其他固定规则
- `poseValid`：姿态是否合理
- `occlusionDetected`：是否存在明显遮挡
- `message`：检测说明，供主业务后端记录或提示用户

#### 失败返回示例
```json
{
  "success": false,
  "message": "Image not found",
  "data": null
}
```

---

### 10.3 生成证件照接口

#### 请求
`POST /ai/generate-id-photo`

#### 请求体
```json
{
  "imageId": "img_001",
  "sourceType": "scene",
  "sceneKey": "passport",
  "customWidthMm": null,
  "customHeightMm": null,
  "backgroundColor": "white",
  "beautyEnabled": false,
  "printLayoutType": "six",
  "originalImagePath": "uploads/original/img_001.jpg"
}
```

#### 字段说明
- `imageId`：图片唯一标识
- `sourceType`：`scene` 或 `custom`
- `sceneKey`：固定场景模板 key，`sourceType=scene` 时必填
- `customWidthMm`：自定义宽度，`sourceType=custom` 时必填
- `customHeightMm`：自定义高度，`sourceType=custom` 时必填
- `backgroundColor`：背景颜色
- `beautyEnabled`：是否开启轻量增强
- `printLayoutType`：排版类型，可为空
- `originalImagePath`：原图路径

#### 成功返回
```json
{
  "success": true,
  "message": "Generate success",
  "data": {
    "imageId": "img_001",
    "previewUrl": "uploads/preview/img_001_preview.jpg",
    "hdUrl": "uploads/hd/img_001_hd.jpg",
    "printUrl": "uploads/print/img_001_print_6.jpg",
    "backgroundColor": "white",
    "widthMm": 33,
    "heightMm": 48,
    "pixelWidth": 413,
    "pixelHeight": 579,
    "qualityStatus": "passed"
  }
}
```

#### data 字段说明
- `imageId`：请求中的 imageId 原样返回
- `previewUrl`：预览图路径，必须返回
- `hdUrl`：高清图路径，必须返回
- `printUrl`：排版图路径，可为空
- `backgroundColor`：最终背景色
- `widthMm`：最终宽度（毫米）
- `heightMm`：最终高度（毫米）
- `pixelWidth`：最终像素宽度
- `pixelHeight`：最终像素高度
- `qualityStatus`：质量状态，枚举值 `passed | warning | failed`

#### 说明
该返回结构必须可被主业务后端直接映射到 `image_results` 表。

#### image_results 映射建议
- `preview_url <- previewUrl`
- `hd_url <- hdUrl`
- `print_url <- printUrl`
- `background_color <- backgroundColor`
- `width_mm <- widthMm`
- `height_mm <- heightMm`
- `pixel_width <- pixelWidth`
- `pixel_height <- pixelHeight`
- `quality_status <- qualityStatus`

#### 失败返回示例
```json
{
  "success": false,
  "message": "Invalid sceneKey",
  "data": null
}
```

---

### 10.4 生成排版图接口

#### 请求
`POST /ai/generate-print-layout`

#### 请求体
```json
{
  "imageId": "img_001",
  "hdImagePath": "uploads/hd/img_001_hd.jpg",
  "layoutType": "six"
}
```

#### 字段说明
- `imageId`：图片唯一标识
- `hdImagePath`：高清图路径
- `layoutType`：排版类型，`six | eight | twelve`

#### 成功返回
```json
{
  "success": true,
  "message": "Generate print layout success",
  "data": {
    "imageId": "img_001",
    "layoutType": "six",
    "printUrl": "uploads/print/img_001_print_6.jpg"
  }
}
```

#### 失败返回示例
```json
{
  "success": false,
  "message": "Invalid layoutType",
  "data": null
}
```

---

## 11. 错误码与错误语义建议

当前第一版可不强制单独 errorCode 字段，但建议 message 稳定且可识别。

常见错误语义：

- `Image not found`
- `Invalid sourceType`
- `Invalid sceneKey`
- `Invalid backgroundColor`
- `Invalid custom size`
- `Invalid layoutType`
- `Face not detected`
- `Multiple faces detected`
- `Image quality too low`
- `Failed to segment portrait`
- `Failed to generate id photo`
- `Failed to generate print layout`

如果后续要扩展 `errorCode`，建议在不破坏当前返回结构前提下增加。

---

## 12. server 侧调用建议

### 12.1 推荐封装方式
主业务后端应封装统一 AI Client，例如：

- `detectPhoto(payload)`
- `generateIdPhoto(payload)`
- `generatePrintLayout(payload)`
- `checkAiHealth()`

### 12.2 调用失败处理
若 AI 服务：
- 超时
- 返回 success=false
- 返回字段不完整

则 `server` 应：
- 标记任务失败
- 记录错误日志
- 返回前端统一错误提示

### 12.3 超时建议
- detect：3~5 秒
- generate-id-photo：10~20 秒
- generate-print-layout：5~10 秒

---

## 13. server 侧任务与结果处理建议

### 13.1 generate-id-photo 调用成功后
主业务后端应：
1. 更新 `image_tasks.status = success`
2. 写入或更新 `image_results`
3. 将返回结果传给前端或供历史记录查询

### 13.2 generate-id-photo 调用失败后
主业务后端应：
1. 更新 `image_tasks.status = failed`
2. 写入 `error_message`
3. 不写入无效结果记录

---

## 14. AI 服务边界说明

AI 服务绝对不负责：

- 用户鉴权
- 订单创建
- 支付回调
- 下载权限判断
- 是否允许查看高清图
- 历史记录查询
- 管理后台统计

这些全部由 `server` 负责。

AI 服务只负责：

- 检测
- 抠图
- 换底
- 裁剪
- 增强
- 排版
- 输出结果文件

---

## 15. 版本兼容要求

第一版上线后，以下内容应尽量保持兼容，不要随意改动：

- API 路径
- 字段名
- 返回结构
- 场景模板 key
- 背景色枚举
- 排版类型枚举
- 输出文件命名规则

如果必须变更，应先更新本契约文档，并同步修改：

- server
- ai-service
- 数据库存储映射逻辑

---

## 16. 推荐联调顺序

1. `GET /ai/health`
2. `POST /ai/detect`
3. `POST /ai/generate-id-photo`
4. `POST /ai/generate-print-layout`

建议先用固定模板场景联调：

- `passport`
- `visa`
- `driver_license`

再测试自定义尺寸：

- `sourceType=custom`

---

## 17. 联调最小样例

### 检测请求
```json
{
  "imageId": "img_demo_001",
  "originalImagePath": "uploads/original/img_demo_001.jpg"
}
```

### 生成请求（固定场景）
```json
{
  "imageId": "img_demo_001",
  "sourceType": "scene",
  "sceneKey": "passport",
  "customWidthMm": null,
  "customHeightMm": null,
  "backgroundColor": "white",
  "beautyEnabled": false,
  "printLayoutType": "six",
  "originalImagePath": "uploads/original/img_demo_001.jpg"
}
```

### 生成请求（自定义尺寸）
```json
{
  "imageId": "img_demo_002",
  "sourceType": "custom",
  "sceneKey": null,
  "customWidthMm": 35,
  "customHeightMm": 45,
  "backgroundColor": "blue",
  "beautyEnabled": false,
  "printLayoutType": "eight",
  "originalImagePath": "uploads/original/img_demo_002.jpg"
}
```

---

## 18. 最终原则

本契约优先级高于 AI 服务内部实现偏好。

也就是说：

- AI 服务必须服从 server 的字段与返回结构约定
- 不允许 AI 服务为了“更优雅”而擅自修改契约
- 所有变更必须先改契约，再改代码
