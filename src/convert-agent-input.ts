import {
  UnsupportedFunctionalityError,
  type LanguageModelV3FilePart,
  type LanguageModelV3Prompt,
  type LanguageModelV3ToolResultOutput,
} from "@ai-sdk/provider"
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils"

export type PerplexityAgentMessageContentPart =
  | { type: "input_text"; text: string }
  | { type: "input_image"; image_url: string }

export type PerplexityAgentInputItem =
  | {
      type: "message"
      role: "user" | "assistant" | "system" | "developer"
      content: PerplexityAgentMessageContentPart[]
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
      output: string
      name?: string
    }

export function convertToPerplexityAgentInput(prompt: LanguageModelV3Prompt): {
  instructions: string
  input: PerplexityAgentInputItem[]
} {
  const systemTexts: string[] = []
  const input: PerplexityAgentInputItem[] = []

  for (const message of prompt) {
    if (message.role === "system") {
      systemTexts.push(message.content)
      continue
    }

    if (message.role === "user") {
      const content: PerplexityAgentMessageContentPart[] = []
      for (const part of message.content) {
        if (part.type === "text") {
          content.push({ type: "input_text", text: part.text })
        } else {
          content.push(convertImagePart(part))
        }
      }
      input.push({ type: "message", role: "user", content })
      continue
    }

    if (message.role === "assistant") {
      let textBuffer: PerplexityAgentMessageContentPart[] = []
      const flushText = (): void => {
        if (textBuffer.length === 0) return
        input.push({
          type: "message",
          role: "assistant",
          content: textBuffer,
        })
        textBuffer = []
      }

      for (const part of message.content) {
        if (part.type === "text") {
          textBuffer.push({ type: "input_text", text: part.text })
        } else if (part.type === "file") {
          textBuffer.push(convertImagePart(part))
        } else if (part.type === "tool-call") {
          flushText()
          input.push({
            type: "function_call",
            call_id: part.toolCallId,
            name: part.toolName,
            arguments:
              typeof part.input === "string"
                ? part.input
                : JSON.stringify(part.input ?? {}),
          })
        } else if (part.type === "tool-result") {
          flushText()
          input.push({
            type: "function_call_output",
            call_id: part.toolCallId,
            name: part.toolName,
            output: toolResultOutputToString(part.output),
          })
        }
      }
      flushText()
      continue
    }

    if (message.role === "tool") {
      for (const part of message.content) {
        if (part.type === "tool-result") {
          input.push({
            type: "function_call_output",
            call_id: part.toolCallId,
            name: part.toolName,
            output: toolResultOutputToString(part.output),
          })
        }
      }
      continue
    }
  }

  return { instructions: systemTexts.join("\n\n"), input }
}

function convertImagePart(
  part: LanguageModelV3FilePart,
): PerplexityAgentMessageContentPart {
  if (!part.mediaType.startsWith("image/")) {
    throw new UnsupportedFunctionalityError({
      functionality: `file parts of media type ${part.mediaType}`,
      message:
        "Perplexity Agent API only supports input_text and input_image content parts.",
    })
  }
  return {
    type: "input_image",
    image_url: dataContentToUrl(part.data, part.mediaType),
  }
}

function dataContentToUrl(
  data: string | Uint8Array | URL,
  mediaType: string,
): string {
  if (data instanceof URL) return data.toString()
  if (data instanceof Uint8Array) {
    return `data:${mediaType};base64,${convertUint8ArrayToBase64(data)}`
  }
  if (data.startsWith("http://") || data.startsWith("https://")) return data
  if (data.startsWith("data:")) return data
  return `data:${mediaType};base64,${data}`
}

function toolResultOutputToString(output: LanguageModelV3ToolResultOutput): string {
  switch (output.type) {
    case "text":
    case "error-text":
      return output.value
    case "json":
    case "error-json":
      return JSON.stringify(output.value)
    case "execution-denied":
      return JSON.stringify({
        denied: true,
        reason: output.reason ?? null,
      })
    case "content":
      return output.value
        .map((item) => (item.type === "text" ? item.text : ""))
        .filter(Boolean)
        .join("\n")
  }
}
