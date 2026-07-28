FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

RUN groupadd --gid 10001 vsss && useradd --create-home --uid 10001 --gid 10001 vsss
WORKDIR /workspace
COPY --chown=vsss:vsss tools/cuda_smoke.py /workspace/tools/cuda_smoke.py
USER vsss
CMD ["python", "/workspace/tools/cuda_smoke.py"]