#! /usr/bin/env bash

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": <model_name>,
        "messages":[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Where is New York?"}],
        "max_tokens": 16,
        "temperature": 0
    }'
