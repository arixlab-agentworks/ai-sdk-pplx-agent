import { defineConfig } from "tsdown"

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  clean: true,
  treeshake: true,
  outExtensions: () => ({ js: ".js", dts: ".d.ts" }),
  external: ["@ai-sdk/provider", "@ai-sdk/provider-utils", "zod"],
})
