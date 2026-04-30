import type {
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
} from "@ai-sdk/provider"
import { z } from "zod"

const annotationSchema = z
  .object({
    type: z.string().nullish(),
    url: z.string().nullish(),
    title: z.string().nullish(),
    start_index: z.number().nullish(),
    end_index: z.number().nullish(),
  })
  .loose()

const outputContentItemSchema = z
  .object({
    type: z.string().nullish(),
    text: z.string().nullish(),
    annotations: z.array(annotationSchema).nullish(),
  })
  .loose()

const outputItemSchema = z
  .object({
    type: z.string().nullish(),
    id: z.string().nullish(),
    role: z.string().nullish(),
    status: z.string().nullish(),
    content: z.array(outputContentItemSchema).nullish(),
    // function_call fields
    call_id: z.string().nullish(),
    name: z.string().nullish(),
    arguments: z.string().nullish(),
  })
  .loose()

const choiceSchema = z
  .object({
    finish_reason: z.string().nullish(),
    message: z.object({ content: z.string().nullish() }).loose().nullish(),
  })
  .loose()

const usageSchema = z
  .object({
    input_tokens: z.number().nullish(),
    output_tokens: z.number().nullish(),
    total_tokens: z.number().nullish(),
    input_tokens_details: z
      .object({
        cache_creation_input_tokens: z.number().nullish(),
        cache_read_input_tokens: z.number().nullish(),
      })
      .loose()
      .nullish(),
  })
  .loose()

export const perplexityAgentResponseSchema = z
  .object({
    id: z.string().nullish(),
    model: z.string().nullish(),
    created_at: z.number().nullish(),
    status: z.string().nullish(),
    output_text: z.string().nullish(),
    output: z.array(outputItemSchema).nullish(),
    choices: z.array(choiceSchema).nullish(),
    usage: usageSchema.nullish(),
  })
  .loose()

export type PerplexityAgentResponse = z.infer<
  typeof perplexityAgentResponseSchema
>
export type PerplexityAgentOutputItem = z.infer<typeof outputItemSchema>
export type PerplexityAgentUsage = z.infer<typeof usageSchema>

export function extractContent(
  data: PerplexityAgentResponse,
): LanguageModelV3Content[] {
  const content: LanguageModelV3Content[] = []

  // Sources from annotations
  const sources = new Set<string>()

  if (data.output) {
    const textBuffer: string[] = []
    for (const item of data.output) {
      if (item.type === "function_call" && item.call_id && item.name) {
        if (textBuffer.length > 0) {
          content.push({ type: "text", text: textBuffer.join("\n") })
          textBuffer.length = 0
        }
        content.push({
          type: "tool-call",
          toolCallId: item.call_id,
          toolName: item.name,
          input: item.arguments ?? "{}",
        })
        continue
      }
      // message items (or items without explicit type) — extract any text + annotations.
      for (const part of item.content ?? []) {
        if (part.text) textBuffer.push(part.text)
        for (const ann of part.annotations ?? []) {
          if (ann.url && !sources.has(ann.url)) {
            sources.add(ann.url)
            content.push({
              type: "source",
              sourceType: "url",
              id: ann.url,
              url: ann.url,
              ...(ann.title && { title: ann.title }),
            })
          }
        }
      }
    }
    if (textBuffer.length > 0) {
      content.unshift({ type: "text", text: textBuffer.join("\n") })
    }
  }

  if (content.length > 0) return content

  // Fallbacks for non-Responses-style payloads
  if (data.output_text) {
    return [{ type: "text", text: data.output_text }]
  }
  const choiceText = data.choices?.[0]?.message?.content
  if (choiceText) {
    return [{ type: "text", text: choiceText }]
  }
  return []
}

export function getResponseMetadata(data: PerplexityAgentResponse): {
  id?: string
  modelId?: string
  timestamp?: Date
} {
  return {
    ...(data.id != null && { id: data.id }),
    ...(data.model != null && { modelId: data.model }),
    ...(data.created_at != null && {
      timestamp: new Date(data.created_at * 1000),
    }),
  }
}

export function mapAgentFinishReason(
  reason: string | null | undefined,
  hasToolCalls = false,
): LanguageModelV3FinishReason {
  if (
    hasToolCalls &&
    (reason == null || reason === "stop" || reason === "completed")
  ) {
    return { unified: "tool-calls", raw: reason ?? undefined }
  }
  switch (reason) {
    case "stop":
    case "length":
      return { unified: reason, raw: reason }
    case "completed":
      return { unified: "stop", raw: reason }
    case "failed":
      return { unified: "error", raw: reason }
    case "content_filter":
      return { unified: "content-filter", raw: reason }
    case "tool_calls":
    case "requires_action":
      return { unified: "tool-calls", raw: reason }
    case undefined:
    case null:
      return { unified: "stop", raw: undefined }
    default:
      return { unified: "other", raw: reason }
  }
}

export function mapUsage(usage: PerplexityAgentUsage | null | undefined): {
  inputTokens: {
    total: number | undefined
    noCache: number | undefined
    cacheRead: number | undefined
    cacheWrite: number | undefined
  }
  outputTokens: {
    total: number | undefined
    text: number | undefined
    reasoning: number | undefined
  }
} {
  const inputTotal = usage?.input_tokens ?? undefined
  const cacheRead = usage?.input_tokens_details?.cache_read_input_tokens ?? undefined
  const cacheWrite =
    usage?.input_tokens_details?.cache_creation_input_tokens ?? undefined
  const noCache =
    inputTotal != null
      ? inputTotal - (cacheRead ?? 0) - (cacheWrite ?? 0)
      : undefined
  const outputTotal = usage?.output_tokens ?? undefined
  return {
    inputTokens: {
      total: inputTotal,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: outputTotal,
      text: outputTotal,
      reasoning: undefined,
    },
  }
}
