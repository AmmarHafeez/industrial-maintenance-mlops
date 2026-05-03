FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "industrial_maintenance_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
