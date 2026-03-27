from __future__ import annotations

import os
from pathlib import Path

import requests

MODEL_NAME = 'u2net.onnx'
MODEL_URL = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'


def main() -> None:
    model_home = Path(os.getenv('U2NET_HOME', '/root/.u2net')).resolve()
    model_home.mkdir(parents=True, exist_ok=True)
    model_path = model_home / MODEL_NAME
    if model_path.exists() and model_path.stat().st_size > 0:
        print(f'rembg model already present: {model_path}')
        return

    print(f'downloading rembg model: {MODEL_URL} -> {model_path}')
    with requests.get(MODEL_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with model_path.open('wb') as fp:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)
    print(f'rembg model prepared: {model_path}')


if __name__ == '__main__':
    main()
