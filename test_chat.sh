#!/bin/bash
# test_chat.sh
# Validation 4.4: Verify the model streams text token-by-token

echo "Running chat mode for 50 tokens with prompt 'Hello'"
python3 main.py --chat --prompt "Hello" --tokens 50
