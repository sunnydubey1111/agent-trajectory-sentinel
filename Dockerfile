# CPU-only, network-free reproduction image for the SYNTHETIC study and the
# deterministic tripwires.
#
# Build (full fidelity, incl. GRU/LSTM/TCN torch baselines — matches the
# committed behaviour snapshot bit-for-bit):
#     docker build -t agentwatch-repro .
#
# Build (lean, torch-free — reproduces the ESN/statistical/linear_ar results but
# a REDUCED snapshot without the torch baselines; ~1 GB smaller):
#     docker build --build-arg REPRO_MODE=lean -t agentwatch-repro:lean .
#
# Run the deterministic gate (fast tests + behaviour snapshot check):
#     docker run --rm agentwatch-repro
# Reproduce the full multiseed study (slow on CPU, deterministic):
#     docker run --rm agentwatch-repro py -m derail.experiments.run_multiseed
#
# Everything runs offline; the network/ollama/Gemini paths are intentionally
# absent (they COLLECT new data, they do not reproduce a committed number).

FROM python:3.14-slim

# REPRO_MODE=full installs the CPU torch wheel so the GRU/LSTM/TCN baselines are
# present and the behaviour snapshot matches exactly; REPRO_MODE=lean omits it.
ARG REPRO_MODE=full

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependency layer first for build caching.
COPY requirements-core.lock.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements-core.lock.txt \
 && if [ "$REPRO_MODE" = "full" ]; then \
        python -m pip install torch==2.12.0+cpu \
            --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Source (respecting .dockerignore; results/traces are carried so --check works).
COPY . .

# `python` is the interpreter the project calls `py` on Windows; alias both.
ENV PYTHONPATH=/app

# Default: the deterministic reproduction gate. Fails the build/run loudly if
# either the fast suite or the behaviour snapshot diverges.
CMD ["sh", "-c", "python -m pytest -q -m 'not slow and not network and not ollama' && python -m devtools.behavior_snapshot --check"]
