import {
  UnsupportedFunctionalityError,
  type JSONSchema7,
  type LanguageModelV3CallOptions,
  type SharedV3Warning,
} from "@ai-sdk/provider"

export type PerplexityAgentTool =
  | {
      type: "web_search"
      filters?: Record<string, unknown>
      max_tokens?: number
      max_tokens_per_page?: number
      user_location?: Record<string, unknown>
    }
  | {
      type: "fetch_url"
      max_urls?: number
    }
  | {
      type: "function"
      name: string
      description?: string
      parameters?: JSONSchema7
      strict?: boolean
    }

export const PERPLEXITY_PROVIDER_TOOL_PREFIX = "perplexity-agent."

export function convertToPerplexityAgentTools(
  tools: LanguageModelV3CallOptions["tools"],
): { tools: PerplexityAgentTool[]; warnings: SharedV3Warning[] } {
  const out: PerplexityAgentTool[] = []
  const warnings: SharedV3Warning[] = []

  if (!tools || tools.length === 0) return { tools: out, warnings }

  for (const tool of tools) {
    if (tool.type === "function") {
      out.push({
        type: "function",
        name: tool.name,
        ...(tool.description && { description: tool.description }),
        ...(tool.inputSchema && {
          parameters: tool.inputSchema as JSONSchema7,
        }),
        ...(tool.strict != null && { strict: tool.strict }),
      })
      continue
    }

    if (tool.type === "provider") {
      if (!tool.id.startsWith(PERPLEXITY_PROVIDER_TOOL_PREFIX)) {
        warnings.push({
          type: "other",
          message: `Provider tool '${tool.id}' is not handled by Perplexity Agent provider.`,
        })
        continue
      }
      const toolName = tool.id.slice(PERPLEXITY_PROVIDER_TOOL_PREFIX.length)
      if (toolName === "web_search") {
        out.push({ type: "web_search", ...tool.args })
        continue
      }
      if (toolName === "fetch_url") {
        out.push({ type: "fetch_url", ...tool.args })
        continue
      }
      warnings.push({
        type: "other",
        message: `Unknown Perplexity provider tool '${toolName}'.`,
      })
      continue
    }
  }

  return { tools: out, warnings }
}

export function assertSupportedToolChoice(
  toolChoice: LanguageModelV3CallOptions["toolChoice"],
): SharedV3Warning[] {
  if (!toolChoice || toolChoice.type === "auto") return []
  if (toolChoice.type === "none") {
    return [
      {
        type: "unsupported",
        feature: "toolChoice=none",
        details:
          "Perplexity Agent API has no explicit 'none' toolChoice. Omit tools to disable.",
      },
    ]
  }
  throw new UnsupportedFunctionalityError({
    functionality: `toolChoice.type=${toolChoice.type}`,
    message:
      "Perplexity Agent API does not expose tool_choice; only 'auto' is supported.",
  })
}
