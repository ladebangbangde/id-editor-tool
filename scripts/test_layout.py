import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description='Test /layout endpoint')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--image', required=True)
    parser.add_argument('--size-key', default='one_inch')
    parser.add_argument('--background-color', default='blue')
    args = parser.parse_args()

    image_path = Path(args.image)
    with image_path.open('rb') as fh:
        response = requests.post(
            f'{args.base_url}/layout',
            files={'image': (image_path.name, fh, 'image/jpeg')},
            data={
                'sizeKey': args.size_key,
                'backgroundColor': args.background_color,
                'enhance': 'false',
                'saveOutput': 'true',
                'paper': '6inch',
            },
            timeout=300,
        )
    print(response.status_code)
    print(response.json())


if __name__ == '__main__':
    main()
