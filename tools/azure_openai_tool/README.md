# Azure OpenAI Image Generation and Editing

## Overview

This plugin uses the Azure OpenAI Images API to generate and edit images with GPT-image deployments. It supports both dated Azure endpoints and explicitly configured v1 endpoints. It does not use the Responses API image-generation tool.

## Configure

### 1. Apply for an Azure OpenAI API Key

Please apply for an API Key on the [Azure OpenAI Platform](https://portal.azure.com/#home). This key will be used for all Azure OpenAI image tools.

In addition to the API Key, you will also need the following information to configure the plugin:

- **Deployment Name**: The name assigned to your Azure OpenAI GPT-image deployment. It can differ from the underlying model name.
- **API Base URL**: Your Azure resource URL for dated APIs, or a URL ending in `/openai/v1/` for the v1 Images API.

### 2. Get Azure OpenAI Image tools from Plugin Marketplace

The Azure OpenAI image tools (for example, **Azure OpenAI Image Generate** and **Azure OpenAI Image Edit**) can be found in the **Plugin Marketplace**.  
Please install the tools you need.

### 3. Fill in the configuration in Dify

On the Dify navigation page, click `Tools > [Installed Azure OpenAI Image Tool Name] > Authorize` and fill in the required fields.  
If you have multiple Azure OpenAI deployments, repeat this for each deployment.

The following fields are available for configuration:

| Field | Description | Example |
| --- | --- | --- |
| **API Key** | Your Azure OpenAI API key. | `********************************` |
| **Deployment Name** | Your Azure deployment name, not necessarily the model ID. | `my-image-deployment` |
| **API Base URL** | Resource URL for dated APIs, or `/openai/v1/` URL for v1. Do not add deployment paths or query parameters. | `https://********.openai.azure.com/` or `https://********.openai.azure.com/openai/v1/` |
| **API Version** | Required for dated endpoints. Ignored for v1, where this field can be empty. Verify that the selected dated version supports your deployment and operation. | `2025-04-01-preview` for a dated endpoint |

The URL selects the API mode. Leaving API Version empty does **not** switch a resource URL to v1. The plugin automatically uses `api-version=preview` for the v1 Images API. Configure a v1 endpoint to apply output format and JPEG compression settings consistently.

**Each authorization check generates a test image and may incur Azure charges.**

#### Image settings

Both generation and editing offer the following settings:

| Setting | Options | Default |
| --- | --- | --- |
| Quality | `auto`, `low`, `medium`, `high`, `xhigh`, `max` | `high` |
| Image size | `1024x1024`, `1536x1024`, `1024x1536`, `auto`, `custom` | `1024x1024` |
| Custom size | A `WxH` string, used when Image size is `custom` | Not set |
| Background | `auto`, `opaque`, `transparent` | `auto` |
| Output format (v1) | `png`, `jpeg` | `png` |
| JPEG compression (v1) | Integer from 0 to 100 | `100` |

Choose settings supported by your Azure deployment and API version. For example, `xhigh` and `max` are quality options for GPT Image 2.5 Flare and Sunburst, not for every GPT-image model. The API validates model-specific quality and size limits and returns an error for unsupported settings.

Background `auto` leaves the API default unchanged. Explicit `opaque` and `transparent` values are sent on both v1 and dated endpoints; support depends on the deployment, API version, and operation. Transparent backgrounds require PNG output. On v1 endpoints, JPEG compression can be set to an integer from 0 to 100.

On dated endpoints, the API determines the output format. The compression setting is sent for generation only; editing ignores it. These choices preserve existing dated request behavior.

### 4. Use the tools

You can use the Azure OpenAI Image tools in the following application types:

#### Chatflow / Workflow applications

Both Chatflow and Workflow applications support nodes for the installed Azure OpenAI Image tools (for example, `Azure OpenAI Image Generate` and `Azure OpenAI Image Edit`). After adding a node, fill in the necessary inputs, such as Prompt, with variables referencing user input or previous node outputs. Reference the image output in the End node or subsequent nodes. To request custom dimensions, set `Image size` to `Custom` and provide `custom_size`, for example `2560x1440` if supported by your deployment.

#### Agent applications

Add the desired Azure OpenAI Image tools in the Agent application settings. Then send an image description for generation, or an image plus an edit instruction, to call the appropriate tool. Azure documents variation-style workflows through image editing and inpainting, so this plugin exposes generation and editing tools rather than a separate variation endpoint.
