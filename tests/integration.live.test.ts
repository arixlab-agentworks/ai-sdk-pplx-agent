/**
 * Live integration tests against the real Perplexity Agent API.
 *
 * Skipped automatically unless `PERPLEXITY_API_KEY` is set. Each test makes a
 * real API call (= real money), so keep the surface small and assertions
 * structural rather than exact-match.
 *
 *   PERPLEXITY_API_KEY=pplx-xxx bun run test
 *   PERPLEXITY_API_KEY=pplx-xxx bun run test -- integration.live
 */
import { describe, expect, it } from "vitest"

import { createPerplexityAgent } from "../src/index"

const apiKey = process.env.PERPLEXITY_API_KEY
const live = apiKey ? describe : describe.skip

const TEST_TIMEOUT = 90_000

// Default to the same model the app uses (apps/web/lib/settings.ts).
// Override via PERPLEXITY_LIVE_MODEL — for reasoning models, bump
// max_output_tokens or output may be empty after thinking budget is consumed.
const MODEL =
  process.env.PERPLEXITY_LIVE_MODEL ?? "google/gemini-3-flash-preview"

const provider = createPerplexityAgent({ apiKey })

live("Perplexity Agent live API", () => {
  it(
    "doGenerate returns text content with usage tokens",
    async () => {
      const model = provider(MODEL)
      const result = await model.doGenerate({
        prompt: [
          { role: "system", content: "Reply with the single word 'pong'." },
          { role: "user", content: [{ type: "text", text: "ping" }] },
        ],
        maxOutputTokens: 256,
      })

      const text = result.content
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map((c) => c.text)
        .join("")

      expect(text.length).toBeGreaterThan(0)
      expect(["stop", "length"]).toContain(result.finishReason.unified)
      expect(result.usage.inputTokens.total).toBeGreaterThan(0)
      expect(result.usage.outputTokens.total).toBeGreaterThan(0)
      expect(result.response?.modelId).toBeDefined()
    },
    TEST_TIMEOUT,
  )

  it(
    "doStream streams text-deltas and ends with finish",
    async () => {
      const model = provider(MODEL)
      const { stream } = await model.doStream({
        prompt: [
          { role: "user", content: [{ type: "text", text: "Count: 1, 2, 3." }] },
        ],
        maxOutputTokens: 256,
      })

      const reader = stream.getReader()
      const types: string[] = []
      const deltas: string[] = []
      let finishUnified: string | undefined
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        types.push(value.type)
        if (value.type === "text-delta") deltas.push(value.delta)
        if (value.type === "finish") finishUnified = value.finishReason.unified
      }

      expect(types[0]).toBe("stream-start")
      expect(types).toContain("text-delta")
      expect(types).toContain("finish")
      expect(deltas.join("").length).toBeGreaterThan(0)
      expect(["stop", "length"]).toContain(finishUnified)
    },
    TEST_TIMEOUT,
  )

  it(
    "function tool call: model emits tool-call content",
    async () => {
      const model = provider(MODEL)
      const result = await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "What is the weather in Seoul? Use the get_weather tool.",
              },
            ],
          },
        ],
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Look up the weather for a city.",
            inputSchema: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
        maxOutputTokens: 256,
      })

      const toolCall = result.content.find((c) => c.type === "tool-call")
      // Some models may answer directly without calling the tool — accept either,
      // but if a tool-call appears, validate its shape.
      if (toolCall && toolCall.type === "tool-call") {
        expect(toolCall.toolName).toBe("get_weather")
        expect(typeof toolCall.toolCallId).toBe("string")
        // Input should be a JSON string parseable into an object containing 'city'.
        const parsed = JSON.parse(
          typeof toolCall.input === "string"
            ? toolCall.input
            : JSON.stringify(toolCall.input),
        )
        expect(parsed).toHaveProperty("city")
        expect(result.finishReason.unified).toBe("tool-calls")
      } else {
        // No tool call — at least confirm the call succeeded with text.
        const text = result.content
          .filter((c): c is { type: "text"; text: string } => c.type === "text")
          .map((c) => c.text)
          .join("")
        expect(text.length).toBeGreaterThan(0)
      }
    },
    TEST_TIMEOUT,
  )

  it(
    "multi-turn: real tool call_id round-trips back to the model",
    async () => {
      const model = provider(MODEL)
      const tools = [
        {
          type: "function" as const,
          name: "get_weather",
          description: "Look up the weather in a city.",
          inputSchema: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      ]

      // Phase 1: ask the model to call the tool, capture the real call_id.
      const phase1 = await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Call the get_weather tool for Seoul. Do not answer directly.",
              },
            ],
          },
        ],
        tools,
        maxOutputTokens: 256,
      })

      const toolCall = phase1.content.find((c) => c.type === "tool-call")
      if (!toolCall || toolCall.type !== "tool-call") {
        // The model declined to call the tool — skip phase 2. Multi-turn shape
        // is already covered by the unit test; the goal here is wire-level.
        return
      }

      // Phase 2: feed the tool result back. This is the real multi-turn round-trip.
      const phase2 = await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Call the get_weather tool for Seoul. Do not answer directly.",
              },
            ],
          },
          {
            role: "assistant",
            content: [
              {
                type: "tool-call",
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                input: toolCall.input,
              },
            ],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                output: {
                  type: "json",
                  value: { tempC: 21, conditions: "clear" },
                },
              },
            ],
          },
        ],
        tools,
        maxOutputTokens: 256,
      })

      const text = phase2.content
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map((c) => c.text)
        .join("")
      expect(
        text.length > 0 || phase2.content.some((c) => c.type === "tool-call"),
      ).toBe(true)
    },
    TEST_TIMEOUT,
  )

  it(
    "web_search provider tool: call succeeds (sources optional)",
    async () => {
      const model = provider(MODEL)
      const result = await model.doGenerate({
        prompt: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "What is the latest stable version of Bun? Cite a source.",
              },
            ],
          },
        ],
        tools: [
          {
            type: "provider",
            id: "perplexity-agent.web_search",
            name: "web_search",
            args: {},
          },
        ],
        maxOutputTokens: 512,
      })

      const text = result.content
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map((c) => c.text)
        .join("")
      expect(text.length).toBeGreaterThan(0)
      // Sources are best-effort — the API may not always return URL annotations.
    },
    TEST_TIMEOUT,
  )
})
