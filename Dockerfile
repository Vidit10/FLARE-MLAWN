# Python-simulator part of the project: PPO allocator, attacks, evaluation, notebook.
# (ns-3 / 5G-LENA is not included; see docs/ns3_setup_wsl.md.)
#
#   docker build -t robust-rra .
#   docker run --rm -p 127.0.0.1:8888:8888 --cpus 4 -e RRA_WORKERS=4 \
#              -v "$PWD/results:/workspace/results" robust-rra
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RRA_WORKERS=1

WORKDIR /workspace
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .
EXPOSE 8888
# The notebook server is only reachable through the port mapping above
# (bound to 127.0.0.1 on the host), so no token is set.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", "--notebook-dir=/workspace"]
