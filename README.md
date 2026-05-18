# ovata-A3B-30B

Fine-tuned Qwen3-30B-A3B-Instruct as a tool-calling executor for Jade lang.

The model is a blind executor — it sees tasks and tools, calls tools to complete work, and has no awareness of the Jade runtime wrapping it. Jade's grammar enforcement layer constrains output to valid tool calls at inference time.

**Output format:** GGUF Q4_K_M (llama.cpp)
