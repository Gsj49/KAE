FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.3 pyarrow==19.0.1
COPY common.py worker.py /opt/scorer/
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HOME=/tmp
USER 65534:65534
WORKDIR /work
