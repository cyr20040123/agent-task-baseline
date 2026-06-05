# curl http://localhost:8030/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen3.5-4B",
#     "messages": [
#       {
#         "role": "user",
#         "content": "苹果通常是什么颜色，一句词回答"
#       }
#     ],
#     "temperature": 0.7,
#     "max_tokens": 200,
#     "logprobs": true,
#     "top_logprobs": 0,
#     "prompt_logprobs": 1
#   }'

# curl http://localhost:8030/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen3.5-4B",
#     "messages": [
#       {
#         "role": "user",
#         "content": "苹果通常是什么颜色，一个词回答"
#       }
#     ],
#     "logprobs": true,    
#     "return_token_ids": true,
#     "prompt_logprobs": 1,
#     "chat_template_kwargs": {
#       "enable_thinking": true
#     }
#   }'

# curl http://localhost:8030/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen3.5-4B",
#     "messages": [
#       {
#         "role": "user",
#         "content": "苹果通常是什么颜色，一个词回答"
#       }
#     ],
#     "logprobs": true,    
#     "return_token_ids": true,
#   }'

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
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'