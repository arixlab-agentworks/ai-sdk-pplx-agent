import { z } from "zod"

export const perplexityAgentErrorSchema = z.object({
  error: z
    .object({
      code: z.union([z.string(), z.number()]).nullish(),
      message: z.string().nullish(),
      type: z.string().nullish(),
    })
    .nullish(),
  detail: z.unknown().nullish(),
  message: z.string().nullish(),
})

export type PerplexityAgentErrorData = z.infer<
  typeof perplexityAgentErrorSchema
>

export function errorToMessage(data: PerplexityAgentErrorData): string {
  return (
    data.error?.message ??
    data.error?.type ??
    data.message ??
    (typeof data.detail === "string" ? data.detail : "unknown error")
  )
}
