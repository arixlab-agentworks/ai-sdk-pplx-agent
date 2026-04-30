import {
  NoSuchModelError,
  type LanguageModelV3,
  type ProviderV3,
} from "@ai-sdk/provider"
import {
  type FetchFunction,
  loadApiKey,
  withoutTrailingSlash,
  withUserAgentSuffix,
} from "@ai-sdk/provider-utils"

import { PerplexityAgentLanguageModel } from "./language-model"

const VERSION = "0.1.1"

export interface PerplexityAgentProvider extends ProviderV3 {
  (modelId: string): LanguageModelV3
  languageModel(modelId: string): LanguageModelV3
}

export interface PerplexityAgentProviderSettings {
  /**
   * Base URL for Perplexity API calls. Defaults to `https://api.perplexity.ai/v1`.
   */
  baseURL?: string
  /**
   * API key. Falls back to `PERPLEXITY_API_KEY` env var.
   */
  apiKey?: string
  /**
   * Extra headers to send with every request.
   */
  headers?: Record<string, string>
  /**
   * Custom fetch implementation (e.g. for testing or interception).
   */
  fetch?: FetchFunction
  /**
   * Default ISO 639-1 language preference applied to every call unless
   * overridden via `providerOptions["perplexity-agent"].languagePreference`.
   */
  defaultLanguagePreference?: string
}

export function createPerplexityAgent(
  options: PerplexityAgentProviderSettings = {},
): PerplexityAgentProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ?? "https://api.perplexity.ai/v1"

  const getHeaders = (): Record<string, string | undefined> =>
    withUserAgentSuffix(
      {
        Authorization: `Bearer ${loadApiKey({
          apiKey: options.apiKey,
          environmentVariableName: "PERPLEXITY_API_KEY",
          description: "Perplexity Agent",
        })}`,
        ...options.headers,
      },
      `ai-sdk-pplx-agent/${VERSION}`,
    )

  const createLanguageModel = (modelId: string): LanguageModelV3 =>
    new PerplexityAgentLanguageModel(modelId, {
      provider: "perplexity-agent",
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
      defaultLanguagePreference: options.defaultLanguagePreference,
    })

  const provider = ((modelId: string) =>
    createLanguageModel(modelId)) as PerplexityAgentProvider

  Object.assign(provider, {
    specificationVersion: "v3" as const,
    languageModel: createLanguageModel,
    embeddingModel: (modelId: string): never => {
      throw new NoSuchModelError({ modelId, modelType: "embeddingModel" })
    },
    textEmbeddingModel: (modelId: string): never => {
      throw new NoSuchModelError({ modelId, modelType: "embeddingModel" })
    },
    imageModel: (modelId: string): never => {
      throw new NoSuchModelError({ modelId, modelType: "imageModel" })
    },
  })

  return provider
}

export const perplexityAgent: PerplexityAgentProvider = createPerplexityAgent()
