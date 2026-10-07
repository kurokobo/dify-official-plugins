# Azure OpenAI 图像生成与编辑

## 概述

本插件使用 Azure OpenAI Images API，通过 GPT-image 部署生成和编辑图像。插件支持日期版本端点和明确配置的 v1 端点，不使用 Responses API 的图像生成工具。

## 配置

### 1. 申请 Azure OpenAI API Key

请前往 [Azure OpenAI Platform](https://portal.azure.com/#home) 申请 API Key。这个 Key 将用于所有 Azure OpenAI 图像工具。

除了 API Key 之外，你还需要准备以下信息来配置插件：

- **Deployment Name**：Azure OpenAI GPT-image 部署的自定义名称，可以与实际模型名称不同。
- **API Base URL**：日期版本 API 使用 Azure 资源 URL；v1 Images API 使用以 `/openai/v1/` 结尾的 URL。

### 2. 从插件市场安装 Azure OpenAI 图像工具

你可以在 **Plugin Marketplace** 中找到 Azure OpenAI 图像工具，例如 **Azure OpenAI Image Generate** 和 **Azure OpenAI Image Edit**。  
请安装你需要使用的工具。

### 3. 在 Dify 中填写配置

在 Dify 导航页中，点击 `Tools > [已安装的 Azure OpenAI Image Tool 名称] > Authorize`，然后填写所需字段。  
如果你有多个 Azure OpenAI 部署，请为每个部署分别重复配置。

可配置字段如下：

| 字段 | 说明 | 示例 |
| --- | --- | --- |
| **API Key** | 你的 Azure OpenAI API Key。 | `********************************` |
| **Deployment Name** | Azure 部署名称，不一定与模型 ID 相同。 | `my-image-deployment` |
| **API Base URL** | 日期版本 API 使用资源 URL；v1 使用 `/openai/v1/` URL。不要添加部署路径或查询参数。 | `https://********.openai.azure.com/` 或 `https://********.openai.azure.com/openai/v1/` |
| **API Version** | 日期版本端点必须填写；v1 忽略此字段，可以留空。请确认选定版本支持你的部署及操作。 | 日期版本端点示例：`2025-04-01-preview` |

API 模式由 URL 决定。API Version 留空**不会**将资源 URL 切换到 v1。插件自动为 v1 Images API 使用 `api-version=preview`。如需一致应用输出格式和 JPEG 压缩率设置，请配置 v1 端点。

**每次授权检查都会生成测试图像，并可能产生 Azure 费用。**

#### 图像设置

生成和编辑工具均提供以下设置：

| 设置 | 选项 | 默认值 |
| --- | --- | --- |
| 质量 | `auto`、`low`、`medium`、`high`、`xhigh`、`max` | `high` |
| 图像大小 | `1024x1024`、`1536x1024`、`1024x1536`、`auto`、`custom` | `1024x1024` |
| 自定义尺寸 | `WxH` 格式，仅在图像大小为 `custom` 时使用 | 未设置 |
| 背景 | `auto`、`opaque`、`transparent` | `auto` |
| 输出格式（v1） | `png`、`jpeg` | `png` |
| JPEG 压缩率（v1） | 0 到 100 的整数 | `100` |

请根据 Azure 部署及 API 版本选择支持的设置。例如，`xhigh` 和 `max` 是 GPT Image 2.5 Flare 和 Sunburst 的质量选项，并非所有 GPT-image 模型都支持。模型特定的质量和尺寸限制由 API 验证，不受支持的设置会返回错误。

背景 `auto` 保留 API 的默认行为。明确设置的 `opaque` 和 `transparent` 会发送到 v1 和日期版本端点；支持情况取决于部署、API 版本及操作。透明背景要求 PNG 输出。v1 端点的 JPEG 压缩率可设置为 0 到 100 的整数。

日期版本端点的输出格式由 API 决定。压缩率设置仅发送给生成操作，编辑操作忽略此设置，以保留现有日期版本请求的行为。

### 4. 使用工具

你可以在以下应用类型中使用 Azure OpenAI 图像工具：

#### Chatflow / Workflow 应用

Chatflow 和 Workflow 都支持添加已安装的 Azure OpenAI 图像工具节点，例如 `Azure OpenAI Image Generate` 和 `Azure OpenAI Image Edit`。添加节点后，为必要输入项（如 Prompt）填写引用用户输入或前序节点输出的变量，并在 End 节点或后续节点中引用图像输出。如需指定自定义尺寸，将“图像大小”设置为“自定义”并填写 `custom_size`，例如部署支持时可使用 `2560x1440`。

#### Agent 应用

在 Agent 应用设置中添加所需工具，然后输入图像生成描述，或图像加编辑指令来调用工具。Azure 文档将 variation 风格的能力归入图像编辑与 inpainting 工作流，因此插件提供生成与编辑工具，而不是单独的 variation 端点。
