FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY apps/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r /tmp/requirements.txt

COPY . /workspace
RUN chmod +x /workspace/apps/api/docker-entrypoint.sh

CMD ["/workspace/apps/api/docker-entrypoint.sh"]
