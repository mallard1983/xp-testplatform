#!/bin/sh
# Starts the Ollama server and pulls nomic-embed-text if not already cached.
# Runs as the container entrypoint for the ollama-embed service.

set -e

echo "[ollama-embed] Starting Ollama server..."
ollama serve &
SERVER_PID=$!

echo "[ollama-embed] Waiting for server to be ready..."
until ollama list > /dev/null 2>&1; do
    sleep 2
done
echo "[ollama-embed] Server ready."

if ollama list | grep -q "nomic-embed-text"; then
    echo "[ollama-embed] nomic-embed-text already present."
else
    echo "[ollama-embed] Pulling nomic-embed-text..."
    ollama pull nomic-embed-text
    echo "[ollama-embed] nomic-embed-text ready."
fi

# Keep container alive by waiting on the server process
wait $SERVER_PID
