#!/bin/bash

# Save the response to a file
curl -X POST http://localhost:8080/v1/agent/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-14B-MLX-4bit",
    "messages": [{"role": "user", "content": "list github repos of user yuvrajxms09"}],
    "max_tokens": 20000,
    "stream": false
  }' > github_repos_response.json

echo "Response saved to github_repos_response.json"
echo ""
echo "Full response content:"
if command -v jq &> /dev/null; then
    jq -r '.choices[0].message.content' github_repos_response.json
else
    # Fallback if jq is not available
    python3 -c "
import json
with open('github_repos_response.json', 'r') as f:
    data = json.load(f)
    print(data['choices'][0]['message']['content'])
" 2>/dev/null || echo "Install jq (brew install jq) or Python to format JSON responses"
fi
