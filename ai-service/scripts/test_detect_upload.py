import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description='Test /ai/detect-upload')
    parser.add_argument('image_path', help='Local image path to upload')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--image-id', default='detect_upload_test')
    args = parser.parse_args()

    image_path = Path(args.image_path)
    with image_path.open('rb') as fp:
        response = requests.post(
            f'{args.base_url}/ai/detect-upload',
            files={'file': (image_path.name, fp, 'image/jpeg')},
            data={'imageId': args.image_id},
            timeout=120,
        )
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
