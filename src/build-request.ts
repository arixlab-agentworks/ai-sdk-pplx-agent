import type {
  LanguageModelV3CallOptions,
  SharedV3Warning,
} from "@ai-sdk/provider"

import {
  convertToPerplexityAgentInput,
  type PerplexityAgentInputItem,
} from "./convert-agent-input"
import {
  assertSupportedToolChoice,
  convertToPerplexityAgentTools,
  type PerplexityAgentTool,
} from "./convert-tools"

export const PERPLEXITY_AGENT_PROVIDER_KEY = "perplexity-agent"

export interface PerplexityAgentProviderOptions {
  webSearch?: boolean
  languagePreference?: string
  preset?: string
  models?: string[]
  maxSteps?: number
}

export interface PerplexityAgentRequestBody {
  model: string
  models?: string[]
  preset?: string
  instructions?: string
  input: PerplexityAgentInputItem[]
  language_preference?: string
  tools?: PerplexityAgentTool[]
  max_output_tokens?: number
  max_steps?: number
  stream?: boolean
}

export interface BuildBodyConfig {
  modelId: string
  defaultLanguagePreference?: string
}

export function buildAgentRequestBody(
  options: LanguageModelV3CallOptions,
  config: BuildBodyConfig,
  extra: { stream?: boolean } = {},
): { body: PerplexityAgentRequestBody; warnings: SharedV3Warning[] } {
  const providerOptions = options.providerOptions?.[
    PERPLEXITY_AGENT_PROVIDER_KEY
  ] as PerplexityAgentProviderOptions | undefined

  const { instructions, input } = convertToPerplexityAgentInput(options.prompt)

  const { tools: convertedTools, warnings: toolWarnings } =
    convertToPerplexityAgentTools(options.tools)
  const choiceWarnings = assertSupportedToolChoice(options.toolChoice)

  // Backward-compatible: providerOptions.webSearch turns on web_search if no tool was set explicitly.
  const tools = [...convertedTools]
  if (
    providerOptions?.webSearch &&
    !tools.some((t) => t.type === "web_search")
  ) {
    tools.push({ type: "web_search" })
  }

  const languagePreference =
    providerOptions?.languagePreference ?? config.defaultLanguagePreference

  const warnings: SharedV3Warning[] = [...toolWarnings, ...choiceWarnings]

  // Surface unsupported sampling settings as warnings rather than failing.
  if (options.temperature != null) {
    warnings.push({
      type: "unsupported",
      feature: "temperature",
      details: "Perplexity Agent API does not accept a temperature parameter.",
    })
  }
  if (options.topP != null) {
    warnings.push({
      type: "unsupported",
      feature: "topP",
    })
  }
  if (options.topK != null) {
    warnings.push({ type: "unsupported", feature: "topK" })
  }
  if (options.frequencyPenalty != null) {
    warnings.push({ type: "unsupported", feature: "frequencyPenalty" })
  }
  if (options.presencePenalty != null) {
    warnings.push({ type: "unsupported", feature: "presencePenalty" })
  }
  if (options.seed != null) {
    warnings.push({ type: "unsupported", feature: "seed" })
  }
  if (options.stopSequences && options.stopSequences.length > 0) {
    warnings.push({ type: "unsupported", feature: "stopSequences" })
  }
  if (options.responseFormat && options.responseFormat.type !== "text") {
    warnings.push({
      type: "unsupported",
      feature: "responseFormat=json",
      details: "Use the response_format provider option once it is wired.",
    })
  }

  const body: PerplexityAgentRequestBody = {
    model: config.modelId,
    ...(providerOptions?.models && { models: providerOptions.models }),
    ...(providerOptions?.preset && { preset: providerOptions.preset }),
    ...(instructions && { instructions }),
    input,
    ...(options.maxOutputTokens != null && {
      max_output_tokens: options.maxOutputTokens,
    }),
    ...(providerOptions?.maxSteps != null && {
      max_steps: providerOptions.maxSteps,
    }),
    ...(languagePreference && { language_preference: languagePreference }),
    ...(tools.length > 0 && { tools }),
    ...(extra.stream && { stream: true }),
  }

  return { body, warnings }
}
