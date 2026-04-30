import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider"
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  type FetchFunction,
  postJsonToApi,
} from "@ai-sdk/provider-utils"

import { buildAgentRequestBody } from "./build-request"
import { errorToMessage, perplexityAgentErrorSchema } from "./error"
import {
  extractContent,
  getResponseMetadata,
  mapAgentFinishReason,
  mapUsage,
  perplexityAgentResponseSchema,
} from "./response"
import {
  createStreamTransformer,
  perplexityStreamEventSchema,
} from "./stream"

export {
  PERPLEXITY_AGENT_PROVIDER_KEY,
  type PerplexityAgentProviderOptions,
} from "./build-request"

export interface PerplexityAgentLanguageModelConfig {
  provider: string
  baseURL: string
  headers: () => Record<string, string | undefined>
  fetch?: FetchFunction
  defaultLanguagePreference?: string
}

export class PerplexityAgentLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"
  readonly modelId: string
  readonly supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^https?:\/\/.+/i],
  }

  private readonly config: PerplexityAgentLanguageModelConfig

  constructor(modelId: string, config: PerplexityAgentLanguageModelConfig) {
    this.modelId = modelId
    this.config = config
  }

  get provider(): string {
    return this.config.provider
  }

  async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    const { body, warnings } = buildAgentRequestBody(options, {
      modelId: this.modelId,
      defaultLanguagePreference: this.config.defaultLanguagePreference,
    })

    const {
      responseHeaders,
      value: response,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: `${this.config.baseURL}/agent`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: perplexityAgentErrorSchema,
        errorToMessage,
      }),
      successfulResponseHandler: createJsonResponseHandler(
        perplexityAgentResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    const content = extractContent(response)
    const hasToolCalls = content.some((c) => c.type === "tool-call")
    const finishReason = mapAgentFinishReason(
      response.choices?.[0]?.finish_reason ?? response.status ?? null,
      hasToolCalls,
    )

    return {
      content,
      finishReason,
      usage: mapUsage(response.usage),
      request: { body },
      response: {
        ...getResponseMetadata(response),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    }
  }

  async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const { body, warnings } = buildAgentRequestBody(
      options,
      {
        modelId: this.modelId,
        defaultLanguagePreference: this.config.defaultLanguagePreference,
      },
      { stream: true },
    )

    const { value: stream, responseHeaders } = await postJsonToApi({
      url: `${this.config.baseURL}/agent`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: perplexityAgentErrorSchema,
        errorToMessage,
      }),
      successfulResponseHandler: createEventSourceResponseHandler(
        perplexityStreamEventSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    const transformed = stream.pipeThrough(createStreamTransformer(warnings))

    return {
      stream: transformed,
      request: { body },
      response: { headers: responseHeaders },
    }
  }
}
