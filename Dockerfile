# perception-mcp-server container image.
#
# Builds a CPU-only image that runs the MCP server with HTTP transport
# on port 8003. The server connects to an external rosbridge_websocket
# and an external Grounding-DINO + SAM HTTP backend — see README and
# docs/ARCHITECTURE.md for the surrounding stack.
#
# Build:
#     docker build -t perception-mcp-server:latest .
#
# Run (pointing at host's rosbridge on localhost:9090):
#     docker run --rm -p 8003:8003 \
#         -e ROSBRIDGE_IP=host.docker.internal \
#         -e ROSBRIDGE_PORT=9090 \
#         -e SAM3_REMOTE_URL=http://your-sam3-host:8001 \
#         perception-mcp-server:latest
#
# On Linux hosts where host.docker.internal does not resolve, either
# use --network host or pass the host's IP explicitly as ROSBRIDGE_IP.

FROM python:3.10-slim

# Install minimal system deps (opencv-python brings most of its own).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
# Copy only pyproject.toml and the package skeleton so a dep change
# doesn't invalidate the build when only source code changes.
COPY pyproject.toml ./
COPY src/ ./src/
COPY server.py ./

# Install the package in editable mode. The runtime deps come from
# pyproject.toml's [project] dependencies block.
RUN pip install --no-cache-dir -e .

# Defaults — override with -e at run time.
ENV ROSBRIDGE_IP=127.0.0.1 \
    ROSBRIDGE_PORT=9090 \
    PYTHONUNBUFFERED=1

# HTTP transport port.
EXPOSE 8003

# Run the MCP server with HTTP transport so external clients can reach
# it on the exposed port. Override CMD to use stdio transport if your
# MCP client connects via the docker-exec stdio pipe instead.
ENTRYPOINT ["python", "server.py"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8003"]
