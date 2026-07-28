import json

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    x = torch.arange(4096, dtype=torch.float32, device="cuda")
    result = (x * x).sum()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "result": float(result),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
