# ai-sdk-pplx-agent

[![build](https://github.com/PleahMaCaka/ai-sdk-pplx-agent/actions/workflows/build.yml/badge.svg)](https://github.com/PleahMaCaka/ai-sdk-pplx-agent/actions/workflows/build.yml)
[![npm version](https://img.shields.io/npm/v/ai-sdk-pplx-agent.svg)](https://www.npmjs.com/package/ai-sdk-pplx-agent)
[![license](https://img.shields.io/npm/l/ai-sdk-pplx-agent.svg)](./LICENSE)

A custom [Vercel AI SDK](https://ai-sdk.dev) provider that re-routes calls through Perplexity's
[`/v1/agent`](https://docs.perplexity.ai/api-reference/agent-post) endpoint, giving you access to
**non-Sonar models** (Anthropic, OpenAI, Google, xAI, NVIDIA, …) with one Perplexity API key.

Implements `LanguageModelV3` / `ProviderV3` and follows the AI SDK
[community provider conventions](https://ai-sdk.dev/providers/community-providers/custom-providers)
(`postJsonToApi` + Zod response schemas + `loadApiKey` env fallback + `withUserAgentSuffix`).

## Why

The official [`@ai-sdk/perplexity`](https://www.npmjs.com/package/@ai-sdk/perplexity) provider only
talks to Perplexity's chat-completions endpoint, which is **Sonar-only**. Perplexity's Agent API
fronts a much broader catalog of third-party models (and adds web search + citations on top). This
package implements `LanguageModelV3` against `/v1/agent` so you can keep using the AI SDK
ergonomics (`generateText`, `providerOptions`, …) while picking from the agent model catalog.

## Install

```bash
bun add ai ai-sdk-pplx-agent
# or
pnpm add ai ai-sdk-pplx-agent
```

`@ai-sdk/provider`, `@ai-sdk/provider-utils`, and `zod` are bundled as runtime deps.

## Usage

```ts
import { generateText } from "ai";
import { createPerplexityAgent } from "ai-sdk-pplx-agent";

const perplexity = createPerplexityAgent({
  // apiKey: process.env.PERPLEXITY_API_KEY  // auto-loaded if omitted
});

const { text } = await generateText({
  model: perplexity("google/gemini-3-flash-preview"),
  system: "You answer in 1 sentence.",
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "What does this label say?" },
        { type: "image", image: "data:image/png;base64,..." },
      ],
    },
  ],
  providerOptions: {
    "perplexity-agent": {
      webSearch: true,
      languagePreference: "ko",
    },
  },
});
```

The provider follows the standard AI SDK shape: `perplexity(modelId)` is a shortcut for
`perplexity.languageModel(modelId)`. `embeddingModel` / `imageModel` throw `NoSuchModelError`.

A pre-built `perplexityAgent` instance is also exported for the common case where you only need
the env-var-driven defaults:

```ts
import { perplexityAgent } from "ai-sdk-pplx-agent";
const model = perplexityAgent("anthropic/claude-sonnet-4-6");
```

### Provider settings

`createPerplexityAgent(options)`:

| field                       | type                     | default                                 |
| --------------------------- | ------------------------ | --------------------------------------- |
| `apiKey`                    | `string`                 | `process.env.PERPLEXITY_API_KEY`        |
| `baseURL`                   | `string`                 | `"https://api.perplexity.ai/v1"`        |
| `headers`                   | `Record<string, string>` | extra headers merged into every request |
| `fetch`                     | `FetchFunction`          | for testing / interception              |
| `defaultLanguagePreference` | `string`                 | applied per-call unless overridden      |

### Per-call provider options

Pass under the `"perplexity-agent"` key in `providerOptions`:

| field                | type       | maps to                                        |
| -------------------- | ---------- | ---------------------------------------------- |
| `webSearch`          | `boolean`  | appends `{ type: "web_search" }` to `tools`    |
| `languagePreference` | `string`   | `language_preference`                          |
| `preset`             | `string`   | `preset` (e.g. `fast-search`, `deep-research`) |
| `models`             | `string[]` | `models` fallback chain (max 5)                |
| `maxSteps`           | `number`   | `max_steps` (1–10) for the research loop       |

The standard `maxOutputTokens` from `generateText` is forwarded as `max_output_tokens`. Pass either
`model` (when constructing the provider) or `preset` / `models` via `providerOptions` — the API
requires at least one.

### Tools

The provider accepts both AI SDK function tools and provider-namespaced tools for the agent
endpoint's built-ins:

```ts
import { generateText, tool } from "ai";
import { perplexityAgent } from "ai-sdk-pplx-agent";
import { z } from "zod";

await generateText({
  model: perplexityAgent("anthropic/claude-sonnet-4-6"),
  tools: {
    // Standard function tool — sent to the model as `{ type: "function", name, parameters }`
    get_weather: tool({
      description: "Look up the weather in a city",
      inputSchema: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ tempC: 21, city }),
    }),
    // Provider-namespaced tools — passed through as native Perplexity tools
    web_search: {
      type: "provider-defined",
      id: "perplexity-agent.web_search",
      args: { max_tokens: 1024 },
    },
    fetch_url: {
      type: "provider-defined",
      id: "perplexity-agent.fetch_url",
      args: { max_urls: 5 },
    },
  },
});
```

Provider tool IDs follow `perplexity-agent.<tool>`; currently `web_search` and `fetch_url` are
recognised. Unknown provider tool IDs are dropped with a warning rather than failing the call.

### Streaming

`streamText` works against `/v1/agent` over SSE. Text deltas, function-call tool invocations,
URL annotations (as `source` parts), and stream-level errors are surfaced as standard
`LanguageModelV3StreamPart`s.

```ts
import { streamText } from "ai";
import { perplexityAgent } from "ai-sdk-pplx-agent";

const { textStream } = streamText({
  model: perplexityAgent("openai/gpt-5.5"),
  prompt: "Tell me a story",
});
for await (const chunk of textStream) process.stdout.write(chunk);
```

### Multi-turn conversations

`assistant` and `tool` messages from the AI SDK are mapped to Perplexity's `function_call` and
`function_call_output` input items, so back-and-forth tool flows round-trip correctly.

## Available models

Perplexity Agent currently routes to the following third-party model IDs (snapshot —
[official list](https://docs.perplexity.ai/docs/agent-api/models)):

| Provider       | Models                                                                                                                                                                            |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI**     | `openai/gpt-5.5`, `openai/gpt-5.4`, `openai/gpt-5.4-mini`, `openai/gpt-5.4-nano`, `openai/gpt-5.2`, `openai/gpt-5.1`, `openai/gpt-5-mini`                                         |
| **Anthropic**  | `anthropic/claude-opus-4-7`, `anthropic/claude-opus-4-6`, `anthropic/claude-opus-4-5`, `anthropic/claude-sonnet-4-6`, `anthropic/claude-sonnet-4-5`, `anthropic/claude-haiku-4-5` |
| **Google**     | `google/gemini-3.1-pro-preview`, `google/gemini-3-flash-preview`                                                                                                                  |
| **xAI**        | `xai/grok-4.20-reasoning`, `xai/grok-4-1-fast-non-reasoning`                                                                                                                      |
| **NVIDIA**     | `nvidia/nemotron-3-super-120b-a12b`                                                                                                                                               |
| **Perplexity** | `perplexity/sonar` (Sonar models also work, but for Sonar-only flows you may prefer `@ai-sdk/perplexity` directly)                                                                |

The list shifts; fetch `GET /v1/models` (with your `PERPLEXITY_API_KEY`) to see what's currently
live for your account.

## Limitations

- **Image-only file parts.** Per the [Perplexity Agent API spec][agent-api], message content only
  supports `input_text` and `input_image` — `input_file` / `input_audio` do not exist. Non-image
  file parts throw `UnsupportedFunctionalityError`. For PDFs or web pages, fetch the URL via the
  `fetch_url` tool instead of attaching as a file.
- **`toolChoice: 'required' | 'tool'` not supported.** The Agent API has no `tool_choice` field;
  only `'auto'` is honored. Other modes throw `UnsupportedFunctionalityError`. `'none'` emits a
  warning (omit `tools` instead).
- **Sampling controls (`temperature`, `topP`, `topK`, penalties, `seed`, `stopSequences`, JSON
  `responseFormat`) emit warnings.** The Agent API does not accept them; pass them anyway and the
  provider will drop them and surface a `SharedV3Warning`.

[agent-api]: https://docs.perplexity.ai/api-reference/agent-post

## Development

```bash
bun install
bun run test       # vitest (mocked — runs without an API key)
bun run build      # tsdown → dist/
```

### Live integration tests

`tests/integration.live.test.ts` exercises the real `/v1/agent` endpoint and is auto-skipped when
`PERPLEXITY_API_KEY` is unset. Each test makes a real billable API call.

```bash
# bash / zsh
PERPLEXITY_API_KEY=pplx-xxx bun run test

# PowerShell
$env:PERPLEXITY_API_KEY = "pplx-xxx"; bun run test

# cmd.exe
set PERPLEXITY_API_KEY=pplx-xxx && bun run test
```

Override the model via `PERPLEXITY_LIVE_MODEL` (default `google/gemini-3-flash-preview`).

Coverage:

- non-streaming `doGenerate` round-trip + usage tokens
- `doStream` SSE → `text-delta` → `finish`
- function tool call shape (model permitting)
- multi-turn `assistant` + `tool` round-trip
- `web_search` provider tool
