import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description='Test /detect endpoint')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--image', required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    with image_path.open('rb') as fh:
        response = requests.post(f'{args.base_url}/detect', files={'file': (image_path.name, fh, 'image/jpeg')}, timeout=120)
    print(response.status_code)
    print(response.json())


if __name__ == '__main__':
    main()
