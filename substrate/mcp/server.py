import os
import httpx

SUBSTRATE_API_URL = os.environ.get("SUBSTRATE_API_URL", "http://substrate-api:8000")

# Stub MCP server — full implementation in Phase 5 (T5.1)
# For now, exposes a health check so the healthcheck in docker-compose passes.

from http.server import HTTPServer, BaseHTTPRequestHandler


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    print("[mcp-server] Stub running on port 8080")
    server.serve_forever()
