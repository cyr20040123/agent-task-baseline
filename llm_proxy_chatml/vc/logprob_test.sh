curl http://localhost:8030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [
      {
        "role": "user",
        "content": "苹果通常是什么颜色，一个词回答"
      }
    ],
    "logprobs": true,    
    "return_token_ids": true,
    "prompt_logprobs": 1
  }'
