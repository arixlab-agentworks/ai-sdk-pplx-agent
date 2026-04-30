import type {
  LanguageModelV3StreamPart,
  SharedV3Warning,
} from "@ai-sdk/provider"
import type { ParseResult } from "@ai-sdk/provider-utils"
import { z } from "zod"

import { mapAgentFinishReason, mapUsage } from "./response"

const usageInStreamSchema = z
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

const annotationSchema = z
  .object({
    type: z.string().nullish(),
    url: z.string().nullish(),
    title: z.string().nullish(),
  })
  .loose()

const streamOutputItemSchema = z
  .object({
    id: z.string().nullish(),
    type: z.string().nullish(),
    role: z.string().nullish(),
    status: z.string().nullish(),
    call_id: z.string().nullish(),
    name: z.string().nullish(),
    arguments: z.string().nullish(),
    content: z
      .array(
        z
          .object({
            type: z.string().nullish(),
            text: z.string().nullish(),
            annotations: z.array(annotationSchema).nullish(),
          })
          .loose(),
      )
      .nullish(),
  })
  .loose()

const streamResponseSchema = z
  .object({
    id: z.string().nullish(),
    model: z.string().nullish(),
    created_at: z.number().nullish(),
    status: z.string().nullish(),
    usage: usageInStreamSchema.nullish(),
    error: z
      .object({
        message: z.string().nullish(),
        code: z.string().nullish(),
        type: z.string().nullish(),
      })
      .loose()
      .nullish(),
  })
  .loose()

export const perplexityStreamEventSchema = z
  .object({
    type: z.string(),
    item_id: z.string().nullish(),
    output_index: z.number().nullish(),
    item: streamOutputItemSchema.nullish(),
    delta: z.string().nullish(),
    text: z.string().nullish(),
    annotation: annotationSchema.nullish(),
    response: streamResponseSchema.nullish(),
  })
  .loose()

export type PerplexityStreamEvent = z.infer<typeof perplexityStreamEventSchema>

interface StreamState {
  /** Active text block ID per item_id (or output_index fallback). */
  textOpen: Map<string, true>
  toolNames: Map<string, string>
  emittedSources: Set<string>
  finishReason: string | null | undefined
  usage: ReturnType<typeof mapUsage> | null
  responseMetaSent: boolean
  hasToolCalls: boolean
}

function newState(): StreamState {
  return {
    textOpen: new Map(),
    toolNames: new Map(),
    emittedSources: new Set(),
    finishReason: undefined,
    usage: null,
    responseMetaSent: false,
    hasToolCalls: false,
  }
}

function textKey(event: PerplexityStreamEvent): string {
  return event.item_id ?? `idx-${event.output_index ?? 0}`
}

export function createStreamTransformer(
  initialWarnings: SharedV3Warning[],
): TransformStream<ParseResult<PerplexityStreamEvent>, LanguageModelV3StreamPart> {
  const state = newState()
  let started = false

  return new TransformStream({
    transform(chunk, controller) {
      if (!started) {
        controller.enqueue({ type: "stream-start", warnings: initialWarnings })
        started = true
      }

      if (!chunk.success) {
        controller.enqueue({ type: "error", error: chunk.error })
        return
      }
      handleEvent(chunk.value, state, controller)
    },
    flush(controller) {
      if (!started) {
        controller.enqueue({ type: "stream-start", warnings: initialWarnings })
      }
      // If we never saw a finish event, emit a minimal one so consumers don't hang.
      if (state.usage == null && state.finishReason === undefined) {
        controller.enqueue({
          type: "finish",
          usage: emptyUsage(),
          finishReason: { unified: "stop", raw: undefined },
        })
      }
    },
  })
}

function handleEvent(
  event: PerplexityStreamEvent,
  state: StreamState,
  controller: TransformStreamDefaultController<LanguageModelV3StreamPart>,
): void {
  switch (event.type) {
    case "response.created":
    case "response.in_progress": {
      const meta = event.response
      if (!state.responseMetaSent && meta) {
        controller.enqueue({
          type: "response-metadata",
          ...(meta.id != null && { id: meta.id }),
          ...(meta.model != null && { modelId: meta.model }),
          ...(meta.created_at != null && {
            timestamp: new Date(meta.created_at * 1000),
          }),
        })
        state.responseMetaSent = true
      }
      return
    }

    case "response.output_item.added": {
      const item = event.item
      if (!item) return
      if (item.type === "function_call" && item.call_id && item.name) {
        state.toolNames.set(item.call_id, item.name)
        state.hasToolCalls = true
        controller.enqueue({
          type: "tool-input-start",
          id: item.call_id,
          toolName: item.name,
        })
      }
      return
    }

    case "response.output_item.done": {
      const item = event.item
      if (!item) return
      if (item.type === "function_call" && item.call_id && item.name) {
        controller.enqueue({
          type: "tool-input-end",
          id: item.call_id,
        })
        controller.enqueue({
          type: "tool-call",
          toolCallId: item.call_id,
          toolName: item.name,
          input: item.arguments ?? "{}",
        })
        state.hasToolCalls = true
      }
      // Emit any annotations on completed message items as sources.
      if (item.type === "message" && item.content) {
        for (const part of item.content) {
          for (const ann of part.annotations ?? []) {
            if (ann.url && !state.emittedSources.has(ann.url)) {
              state.emittedSources.add(ann.url)
              controller.enqueue({
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
      return
    }

    case "response.output_text.delta": {
      const key = textKey(event)
      if (!state.textOpen.has(key)) {
        controller.enqueue({ type: "text-start", id: key })
        state.textOpen.set(key, true)
      }
      const delta = event.delta ?? ""
      if (delta) controller.enqueue({ type: "text-delta", id: key, delta })
      return
    }

    case "response.output_text.done": {
      const key = textKey(event)
      if (state.textOpen.has(key)) {
        controller.enqueue({ type: "text-end", id: key })
        state.textOpen.delete(key)
      }
      return
    }

    case "response.completed": {
      const meta = event.response
      if (!state.responseMetaSent && meta) {
        controller.enqueue({
          type: "response-metadata",
          ...(meta.id != null && { id: meta.id }),
          ...(meta.model != null && { modelId: meta.model }),
          ...(meta.created_at != null && {
            timestamp: new Date(meta.created_at * 1000),
          }),
        })
        state.responseMetaSent = true
      }
      // Close any still-open text blocks.
      for (const key of state.textOpen.keys()) {
        controller.enqueue({ type: "text-end", id: key })
      }
      state.textOpen.clear()

      const usage = mapUsage(meta?.usage)
      state.usage = usage
      state.finishReason = meta?.status ?? null
      controller.enqueue({
        type: "finish",
        usage,
        finishReason: mapAgentFinishReason(
          meta?.status ?? null,
          state.hasToolCalls,
        ),
      })
      return
    }

    case "response.failed": {
      const meta = event.response
      controller.enqueue({
        type: "error",
        error: meta?.error ?? { message: "Perplexity Agent stream failed" },
      })
      controller.enqueue({
        type: "finish",
        usage: emptyUsage(),
        finishReason: { unified: "error", raw: meta?.status ?? undefined },
      })
      state.finishReason = meta?.status ?? "error"
      return
    }

    default:
      return
  }
}

function emptyUsage(): ReturnType<typeof mapUsage> {
  return mapUsage(null)
}
