FROM ollama/ollama:0.6.8

ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_KEEP_ALIVE=-1

COPY apps/ollama/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
RUN /bin/sh -c "ollama serve > /tmp/ollama-build.log 2>&1 & \
  pid=$!; \
  sleep 8; \
  ollama pull tinyllama; \
  kill $pid; \
  wait $pid || true"

CMD ["/entrypoint.sh"]
