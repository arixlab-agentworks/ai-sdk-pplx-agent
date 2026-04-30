import {
  APICallError,
  NoSuchModelError,
  UnsupportedFunctionalityError,
  type JSONValue,
  type LanguageModelV3,
  type LanguageModelV3CallOptions,
} from "@ai-sdk/provider"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  createPerplexityAgent,
  PERPLEXITY_AGENT_PROVIDER_KEY,
  PerplexityAgentLanguageModel,
  type PerplexityAgentProvider,
} from "../src/index"

type AgentInputPart =
  | { type: "input_text"; text: string }
  | { type: "input_image"; image_url: string }

type AgentInputItem =
  | {
      type: "message"
      role: "user" | "assistant" | "system" | "developer"
      content: AgentInputPart[]
    }
  | {
      type: "function_call"
      call_id: string
      name: string
      arguments: string
    }
  | {
      type: "function_call_output"
      call_id: string
      name?: string
      output: string
    }

interface AgentRequestBody {
  model: string
  models?: string[]
  preset?: string
  instructions?: string
  input: AgentInputItem[]
  language_preference?: string
  tools?: Array<
    | { type: "web_search"; [k: string]: unknown }
    | { type: "fetch_url"; max_urls?: number }
    | {
        type: "function"
        name: string
        description?: string
        parameters?: Record<string, unknown>
        strict?: boolean
      }
  >
  max_output_tokens?: number
  max_steps?: number
  stream?: boolean
}

interface CapturedRequest {
  url: string
  method: string
  headers: Record<string, string>
  body: AgentRequestBody
  signal?: AbortSignal
}

let captured: CapturedRequest | null
let mockResponse: () => Response

function makeFetch(): typeof fetch {
  return vi.fn(async (input: string | URL, init?: RequestInit) => {
    captured = {
      url: typeof input === "string" ? input : input.toString(),
      method: init?.method ?? "GET",
      headers: init?.headers as Record<string, string>,
      body: init?.body
        ? (JSON.parse(init.body as string) as AgentRequestBody)
        : ({} as AgentRequestBody),
      signal: init?.signal ?? undefined,
    }
    return mockResponse()
  }) as unknown as typeof fetch
}

function jsonResponse(body: JSONValue, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
    ...init,
  })
}

function buildProvider(
  overrides: Partial<{
    apiKey: string
    baseURL: string
    headers: Record<string, string>
  }> = {},
): PerplexityAgentProvider {
  return createPerplexityAgent({
    apiKey: overrides.apiKey ?? "sk-test",
    baseURL: overrides.baseURL,
    headers: overrides.headers,
    fetch: makeFetch(),
  })
}

function buildModel(
  modelId = "google/gemini-3-flash-preview",
): LanguageModelV3 {
  return buildProvider()(modelId)
}

function callOptions(
  prompt: LanguageModelV3CallOptions["prompt"],
  extras: Partial<LanguageModelV3CallOptions> = {},
): LanguageModelV3CallOptions {
  return { prompt, ...extras }
}

function messageItem(
  body: AgentRequestBody | undefined,
  index: number,
): {
  type: "message"
  role: "user" | "assistant" | "system" | "developer"
  content: AgentInputPart[]
} {
  const item = body?.input[index]
  if (!item || item.type !== "message") {
    throw new Error(`expected message item at index ${index}`)
  }
  return item
}

beforeEach(() => {
  captured = null
  mockResponse = () => jsonResponse({ output_text: "ok" })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("PerplexityAgentLanguageModel basics", () => {
  it("exposes spec metadata", () => {
    const model = buildModel("google/gemini-3-flash-preview")
    expect(model.specificationVersion).toBe("v3")
    expect(model.provider).toBe("perplexity-agent")
    expect(model.modelId).toBe("google/gemini-3-flash-preview")
    expect(model.supportedUrls).toEqual({ "image/*": [/^https?:\/\/.+/i] })
  })

  it("uses default Perplexity baseURL when none is provided", async () => {
    const model = buildModel()
    await model.doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.url).toBe("https://api.perplexity.ai/v1/agent")
  })

  it("trims trailing slashes from custom baseURL", async () => {
    const provider = createPerplexityAgent({
      apiKey: "sk-test",
      baseURL: "https://proxy.local/v1/",
      fetch: makeFetch(),
    })
    await provider("test").doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.url).toBe("https://proxy.local/v1/agent")
  })

  it("sends Authorization, Content-Type, and User-Agent suffix", async () => {
    const provider = buildProvider({ apiKey: "sk-abc" })
    await provider("test").doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.headers["authorization"]).toBe("Bearer sk-abc")
    expect(captured?.headers["content-type"]).toBe("application/json")
    expect(captured?.headers["user-agent"]).toMatch(
      /ai-sdk-pplx-agent\/[0-9.]+/,
    )
  })

  it("falls back to PERPLEXITY_API_KEY env var", async () => {
    vi.stubEnv("PERPLEXITY_API_KEY", "sk-env")
    try {
      const provider = createPerplexityAgent({ fetch: makeFetch() })
      await provider("test").doGenerate(
        callOptions([
          { role: "user", content: [{ type: "text", text: "hi" }] },
        ]),
      )
      expect(captured?.headers["authorization"]).toBe("Bearer sk-env")
    } finally {
      vi.unstubAllEnvs()
    }
  })

  it("merges custom provider headers and request-level headers", async () => {
    const provider = buildProvider({ headers: { "X-Provider": "p" } })
    await provider("test").doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        headers: { "X-Trace": "t", "X-Skip": undefined },
      }),
    )
    expect(captured?.headers["x-provider"]).toBe("p")
    expect(captured?.headers["x-trace"]).toBe("t")
    expect("x-skip" in (captured?.headers ?? {})).toBe(false)
  })
})

describe("prompt → Perplexity input conversion", () => {
  it("forwards a plain text user message", async () => {
    await buildModel().doGenerate(
      callOptions([
        { role: "user", content: [{ type: "text", text: "hello" }] },
      ]),
    )
    expect(captured?.body).toMatchObject({
      input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "hello" }],
        },
      ],
    })
    expect(captured?.body).not.toHaveProperty("instructions")
  })

  it("joins system messages into instructions", async () => {
    await buildModel().doGenerate(
      callOptions([
        { role: "system", content: "be terse" },
        { role: "system", content: "speak korean" },
        { role: "user", content: [{ type: "text", text: "hi" }] },
      ]),
    )
    expect(captured?.body.instructions).toBe("be terse\n\nspeak korean")
  })

  it("converts data URI image file parts to input_image", async () => {
    await buildModel().doGenerate(
      callOptions([
        {
          role: "user",
          content: [
            { type: "text", text: "what is this?" },
            {
              type: "file",
              mediaType: "image/png",
              data: "data:image/png;base64,AAAA",
            },
          ],
        },
      ]),
    )
    expect(messageItem(captured?.body, 0).content).toEqual([
      { type: "input_text", text: "what is this?" },
      { type: "input_image", image_url: "data:image/png;base64,AAAA" },
    ])
  })

  it("wraps bare base64 strings into a data URI", async () => {
    await buildModel().doGenerate(
      callOptions([
        {
          role: "user",
          content: [
            { type: "file", mediaType: "image/jpeg", data: "Zm9vYmFy" },
          ],
        },
      ]),
    )
    expect(messageItem(captured?.body, 0).content[0]).toEqual({
      type: "input_image",
      image_url: "data:image/jpeg;base64,Zm9vYmFy",
    })
  })

  it("base64-encodes Uint8Array image data", async () => {
    await buildModel().doGenerate(
      callOptions([
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "image/png",
              data: new Uint8Array([1, 2, 3, 4]),
            },
          ],
        },
      ]),
    )
    expect(messageItem(captured?.body, 0).content[0]).toEqual({
      type: "input_image",
      image_url: `data:image/png;base64,${Buffer.from(new Uint8Array([1, 2, 3, 4])).toString("base64")}`,
    })
  })

  it("passes URL image parts through verbatim", async () => {
    await buildModel().doGenerate(
      callOptions([
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "image/png",
              data: new URL("https://example.com/cat.png"),
            },
          ],
        },
      ]),
    )
    expect(messageItem(captured?.body, 0).content[0]).toEqual({
      type: "input_image",
      image_url: "https://example.com/cat.png",
    })
  })

  it("rejects non-image file media types", async () => {
    await expect(
      buildModel().doGenerate(
        callOptions([
          {
            role: "user",
            content: [
              {
                type: "file",
                mediaType: "application/pdf",
                data: "data:application/pdf;base64,AAAA",
              },
            ],
          },
        ]),
      ),
    ).rejects.toBeInstanceOf(UnsupportedFunctionalityError)
  })

  it("converts assistant text messages to assistant message items", async () => {
    await buildModel().doGenerate(
      callOptions([
        { role: "user", content: [{ type: "text", text: "hi" }] },
        { role: "assistant", content: [{ type: "text", text: "hello" }] },
      ]),
    )
    expect(captured?.body.input).toEqual([
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "hi" }],
      },
      {
        type: "message",
        role: "assistant",
        content: [{ type: "input_text", text: "hello" }],
      },
    ])
  })

  it("converts assistant tool-call parts to function_call items", async () => {
    await buildModel().doGenerate(
      callOptions([
        { role: "user", content: [{ type: "text", text: "weather please" }] },
        {
          role: "assistant",
          content: [
            { type: "text", text: "let me check" },
            {
              type: "tool-call",
              toolCallId: "call_1",
              toolName: "get_weather",
              input: { city: "Seoul" },
            },
          ],
        },
      ]),
    )
    expect(captured?.body.input).toEqual([
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "weather please" }],
      },
      {
        type: "message",
        role: "assistant",
        content: [{ type: "input_text", text: "let me check" }],
      },
      {
        type: "function_call",
        call_id: "call_1",
        name: "get_weather",
        arguments: JSON.stringify({ city: "Seoul" }),
      },
    ])
  })

  it("converts tool messages to function_call_output items", async () => {
    await buildModel().doGenerate(
      callOptions([
        { role: "user", content: [{ type: "text", text: "ask" }] },
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_x",
              toolName: "lookup",
              input: { id: 1 },
            },
          ],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_x",
              toolName: "lookup",
              output: { type: "json", value: { ok: true, n: 42 } },
            },
          ],
        },
      ]),
    )
    expect(captured?.body.input).toEqual([
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "ask" }],
      },
      {
        type: "function_call",
        call_id: "call_x",
        name: "lookup",
        arguments: JSON.stringify({ id: 1 }),
      },
      {
        type: "function_call_output",
        call_id: "call_x",
        name: "lookup",
        output: JSON.stringify({ ok: true, n: 42 }),
      },
    ])
  })
})

describe("provider options + standard call options", () => {
  it("forwards maxOutputTokens to max_output_tokens", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        maxOutputTokens: 256,
      }),
    )
    expect(captured?.body.max_output_tokens).toBe(256)
  })

  it("uses defaultLanguagePreference from provider settings", async () => {
    const provider = createPerplexityAgent({
      apiKey: "sk-test",
      defaultLanguagePreference: "ko",
      fetch: makeFetch(),
    })
    await provider("test").doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.body.language_preference).toBe("ko")
  })

  it("call-level languagePreference overrides defaultLanguagePreference", async () => {
    const provider = createPerplexityAgent({
      apiKey: "sk-test",
      defaultLanguagePreference: "ko",
      fetch: makeFetch(),
    })
    await provider("test").doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        providerOptions: {
          [PERPLEXITY_AGENT_PROVIDER_KEY]: { languagePreference: "en" },
        },
      }),
    )
    expect(captured?.body.language_preference).toBe("en")
  })

  it("sends languagePreference from providerOptions", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        providerOptions: {
          [PERPLEXITY_AGENT_PROVIDER_KEY]: { languagePreference: "ko" },
        },
      }),
    )
    expect(captured?.body.language_preference).toBe("ko")
  })

  it("omits language_preference when no option is set", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.body).not.toHaveProperty("language_preference")
  })

  it("includes web_search tool when webSearch is true", async () => {
    await buildModel().doGenerate(
      callOptions(
        [{ role: "user", content: [{ type: "text", text: "search" }] }],
        {
          providerOptions: {
            [PERPLEXITY_AGENT_PROVIDER_KEY]: { webSearch: true },
          },
        },
      ),
    )
    expect(captured?.body.tools).toEqual([{ type: "web_search" }])
  })

  it("omits tools when webSearch is false or unset", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.body).not.toHaveProperty("tools")
  })
})

describe("response parsing", () => {
  it("extracts text from output_text", async () => {
    mockResponse = () => jsonResponse({ output_text: "direct text" })
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "x" }] }]),
    )
    expect(result.content).toEqual([{ type: "text", text: "direct text" }])
    expect(result.finishReason.unified).toBe("stop")
  })

  it("falls back to output[].content[].text array", async () => {
    mockResponse = () =>
      jsonResponse({
        output: [
          { content: [{ text: "first" }, { text: "second" }] },
          { content: [{ text: "third" }] },
        ],
      })
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "x" }] }]),
    )
    expect(result.content).toEqual([
      { type: "text", text: "first\nsecond\nthird" },
    ])
  })

  it("falls back to choices[0].message.content", async () => {
    mockResponse = () =>
      jsonResponse({ choices: [{ message: { content: "choice text" } }] })
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "x" }] }]),
    )
    expect(result.content).toEqual([{ type: "text", text: "choice text" }])
  })

  it("returns empty content when response has no text", async () => {
    mockResponse = () => jsonResponse({})
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "x" }] }]),
    )
    expect(result.content).toEqual([])
  })

  it("maps finish_reason and surfaces token usage", async () => {
    mockResponse = () =>
      jsonResponse({
        output_text: "done",
        choices: [{ finish_reason: "length" }],
        usage: { input_tokens: 12, output_tokens: 7, total_tokens: 19 },
      })
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "x" }] }]),
    )
    expect(result.finishReason).toEqual({ unified: "length", raw: "length" })
    expect(result.usage.inputTokens.total).toBe(12)
    expect(result.usage.outputTokens.total).toBe(7)
  })

  it("populates response metadata from id/model/created_at", async () => {
    mockResponse = () =>
      jsonResponse({
        output_text: "ok",
        id: "resp_123",
        model: "google/gemini-3-flash-preview",
        created_at: 1_700_000_000,
      })
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "x" }] }]),
    )
    expect(result.response?.id).toBe("resp_123")
    expect(result.response?.modelId).toBe("google/gemini-3-flash-preview")
    expect(result.response?.timestamp).toEqual(new Date(1_700_000_000 * 1000))
  })
})

describe("error handling", () => {
  it("throws APICallError for non-2xx responses", async () => {
    mockResponse = () =>
      new Response(JSON.stringify({ error: { message: "rate limited" } }), {
        status: 429,
        headers: { "content-type": "application/json" },
      })
    await expect(
      buildModel().doGenerate(
        callOptions([
          { role: "user", content: [{ type: "text", text: "hi" }] },
        ]),
      ),
    ).rejects.toBeInstanceOf(APICallError)
  })

  it("APICallError carries status code", async () => {
    mockResponse = () =>
      new Response(JSON.stringify({ error: { message: "boom" } }), {
        status: 503,
        headers: { "content-type": "application/json" },
      })
    try {
      await buildModel().doGenerate(
        callOptions([
          { role: "user", content: [{ type: "text", text: "hi" }] },
        ]),
      )
      throw new Error("expected throw")
    } catch (err) {
      expect(APICallError.isInstance(err)).toBe(true)
      const e = err as APICallError
      expect(e.statusCode).toBe(503)
    }
  })

  it("propagates abort signal to fetch", async () => {
    const controller = new AbortController()
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "x" }] }], {
        abortSignal: controller.signal,
      }),
    )
    expect(captured?.signal).toBe(controller.signal)
  })

})

describe("createPerplexityAgent factory (ProviderV3 conformance)", () => {
  it("provider is callable as a shortcut for languageModel", () => {
    const provider = buildProvider()
    expect(provider("model-a").modelId).toBe("model-a")
    expect(provider.languageModel("model-b").modelId).toBe("model-b")
  })

  it("declares specificationVersion v3", () => {
    expect(buildProvider().specificationVersion).toBe("v3")
  })

  it("throws NoSuchModelError for embedding/image model lookups", () => {
    const provider = buildProvider()
    expect(() => provider.embeddingModel("any")).toThrow(NoSuchModelError)
    expect(() => provider.imageModel("any")).toThrow(NoSuchModelError)
  })

  it("returns a PerplexityAgentLanguageModel instance", async () => {
    const provider = buildProvider()
    const model = provider("custom-model")
    expect(model).toBeInstanceOf(PerplexityAgentLanguageModel)
    await model.doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.body.model).toBe("custom-model")
  })
})

describe("preset and model fallback chain provider options", () => {
  it("forwards preset from providerOptions", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        providerOptions: {
          [PERPLEXITY_AGENT_PROVIDER_KEY]: { preset: "deep-research" },
        },
      }),
    )
    expect(captured?.body.preset).toBe("deep-research")
  })

  it("forwards models fallback chain from providerOptions", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        providerOptions: {
          [PERPLEXITY_AGENT_PROVIDER_KEY]: {
            models: ["openai/gpt-5.5", "anthropic/claude-opus-4-7"],
          },
        },
      }),
    )
    expect(captured?.body.models).toEqual([
      "openai/gpt-5.5",
      "anthropic/claude-opus-4-7",
    ])
  })

  it("forwards maxSteps from providerOptions", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        providerOptions: {
          [PERPLEXITY_AGENT_PROVIDER_KEY]: { maxSteps: 5 },
        },
      }),
    )
    expect(captured?.body.max_steps).toBe(5)
  })
})

describe("function and provider tools wiring", () => {
  it("converts AI SDK function tools to Perplexity function tools", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Look up the weather",
            inputSchema: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
      }),
    )
    expect(captured?.body.tools).toEqual([
      {
        type: "function",
        name: "get_weather",
        description: "Look up the weather",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      },
    ])
  })

  it("maps perplexity-agent.web_search provider tool with args", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        tools: [
          {
            type: "provider",
            id: "perplexity-agent.web_search",
            name: "web_search",
            args: { max_tokens: 1024 },
          },
        ],
      }),
    )
    expect(captured?.body.tools).toEqual([
      { type: "web_search", max_tokens: 1024 },
    ])
  })

  it("maps perplexity-agent.fetch_url provider tool", async () => {
    await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        tools: [
          {
            type: "provider",
            id: "perplexity-agent.fetch_url",
            name: "fetch_url",
            args: { max_urls: 5 },
          },
        ],
      }),
    )
    expect(captured?.body.tools).toEqual([{ type: "fetch_url", max_urls: 5 }])
  })

  it("warns on unknown provider tools without rejecting the call", async () => {
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }], {
        tools: [
          {
            type: "provider",
            id: "openai.web_search",
            name: "web_search",
            args: {},
          },
        ],
      }),
    )
    expect(captured?.body.tools).toBeUndefined()
    expect(result.warnings).toContainEqual(
      expect.objectContaining({ type: "other" }),
    )
  })

  it("rejects unsupported toolChoice (required/tool) with UnsupportedFunctionalityError", async () => {
    await expect(
      buildModel().doGenerate(
        callOptions(
          [{ role: "user", content: [{ type: "text", text: "hi" }] }],
          {
            toolChoice: { type: "required" },
            tools: [
              {
                type: "function",
                name: "noop",
                inputSchema: { type: "object" },
              },
            ],
          },
        ),
      ),
    ).rejects.toBeInstanceOf(UnsupportedFunctionalityError)
  })

  it("returns tool-call content when response includes function_call output", async () => {
    mockResponse = () =>
      jsonResponse({
        output: [
          {
            type: "function_call",
            id: "fc_1",
            call_id: "call_42",
            name: "get_weather",
            arguments: JSON.stringify({ city: "Seoul" }),
          },
        ],
      })
    const result = await buildModel().doGenerate(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(result.content).toEqual([
      {
        type: "tool-call",
        toolCallId: "call_42",
        toolName: "get_weather",
        input: JSON.stringify({ city: "Seoul" }),
      },
    ])
    expect(result.finishReason.unified).toBe("tool-calls")
  })
})

describe("doStream (SSE)", () => {
  function sseResponse(events: Array<Record<string, unknown>>): Response {
    const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("")
    return new Response(body, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    })
  }

  async function collectStream(
    stream: ReadableStream<unknown>,
  ): Promise<unknown[]> {
    const reader = stream.getReader()
    const out: unknown[] = []
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      out.push(value)
    }
    return out
  }

  it("sets stream:true in the request body", async () => {
    mockResponse = () =>
      sseResponse([
        {
          type: "response.completed",
          response: { id: "r_1", model: "m", status: "completed", usage: {} },
        },
      ])
    const stream = await buildModel().doStream(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    expect(captured?.body.stream).toBe(true)
    await collectStream(stream.stream)
  })

  it("emits stream-start, text deltas, and finish for a basic response", async () => {
    mockResponse = () =>
      sseResponse([
        {
          type: "response.created",
          response: { id: "r_x", model: "m", created_at: 1_700_000_000 },
        },
        { type: "response.output_text.delta", item_id: "msg_1", delta: "hel" },
        { type: "response.output_text.delta", item_id: "msg_1", delta: "lo" },
        { type: "response.output_text.done", item_id: "msg_1", text: "hello" },
        {
          type: "response.completed",
          response: {
            id: "r_x",
            model: "m",
            status: "completed",
            usage: {
              input_tokens: 3,
              output_tokens: 1,
              total_tokens: 4,
            },
          },
        },
      ])
    const stream = await buildModel().doStream(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    const parts = (await collectStream(stream.stream)) as Array<{
      type: string
      [k: string]: unknown
    }>
    const types = parts.map((p) => p.type)
    expect(types[0]).toBe("stream-start")
    expect(types).toContain("text-start")
    expect(types).toContain("text-delta")
    expect(types).toContain("text-end")
    expect(types[types.length - 1]).toBe("finish")

    const deltas = parts
      .filter((p) => p.type === "text-delta")
      .map((p) => p.delta)
    expect(deltas).toEqual(["hel", "lo"])

    const finish = parts[parts.length - 1] as {
      type: "finish"
      finishReason: { unified: string }
      usage: { inputTokens: { total: number } }
    }
    expect(finish.finishReason.unified).toBe("stop")
    expect(finish.usage.inputTokens.total).toBe(3)
  })

  it("emits tool-call stream parts when response.output_item.done is a function_call", async () => {
    mockResponse = () =>
      sseResponse([
        { type: "response.created", response: { id: "r_y" } },
        {
          type: "response.output_item.added",
          item: { type: "function_call", call_id: "c_1", name: "f" },
        },
        {
          type: "response.output_item.done",
          item: {
            type: "function_call",
            call_id: "c_1",
            name: "f",
            arguments: "{\"a\":1}",
          },
        },
        {
          type: "response.completed",
          response: { id: "r_y", status: "completed", usage: {} },
        },
      ])
    const stream = await buildModel().doStream(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    const parts = (await collectStream(stream.stream)) as Array<{
      type: string
      [k: string]: unknown
    }>
    const toolCall = parts.find((p) => p.type === "tool-call")
    expect(toolCall).toMatchObject({
      type: "tool-call",
      toolCallId: "c_1",
      toolName: "f",
      input: '{"a":1}',
    })
    const finish = parts[parts.length - 1] as {
      type: "finish"
      finishReason: { unified: string }
    }
    expect(finish.finishReason.unified).toBe("tool-calls")
  })

  it("emits error stream part on response.failed", async () => {
    mockResponse = () =>
      sseResponse([
        {
          type: "response.failed",
          response: {
            error: { message: "model overloaded" },
            status: "failed",
          },
        },
      ])
    const stream = await buildModel().doStream(
      callOptions([{ role: "user", content: [{ type: "text", text: "hi" }] }]),
    )
    const parts = (await collectStream(stream.stream)) as Array<{
      type: string
      [k: string]: unknown
    }>
    expect(parts.find((p) => p.type === "error")).toBeDefined()
    const finish = parts[parts.length - 1] as {
      type: "finish"
      finishReason: { unified: string }
    }
    expect(finish.finishReason.unified).toBe("error")
  })
})
