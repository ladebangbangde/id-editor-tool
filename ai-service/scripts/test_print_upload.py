import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description='Test /ai/generate-print-layout-upload')
    parser.add_argument('image_path', help='Local image path to upload')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--image-id', default='print_upload_test')
    parser.add_argument('--layout-type', default='six', choices=['six', 'eight', 'twelve'])
    parser.add_argument('--source-type', default='scene', choices=['scene', 'custom'])
    parser.add_argument('--scene-key', default='passport')
    parser.add_argument('--size-name')
    parser.add_argument('--custom-width-mm', type=int)
    parser.add_argument('--custom-height-mm', type=int)
    parser.add_argument('--background-color', default='white', choices=['white', 'blue', 'red'])
    parser.add_argument('--beauty-enabled', action='store_true')
    args = parser.parse_args()

    image_path = Path(args.image_path)
    data = {
        'imageId': args.image_id,
        'layoutType': args.layout_type,
        'sourceType': args.source_type,
        'sceneKey': args.scene_key,
        'backgroundColor': args.background_color,
        'beautyEnabled': str(args.beauty_enabled).lower(),
    }
    if args.size_name:
        data['sizeName'] = args.size_name
    if args.custom_width_mm:
        data['customWidthMm'] = str(args.custom_width_mm)
    if args.custom_height_mm:
        data['customHeightMm'] = str(args.custom_height_mm)

    with image_path.open('rb') as fp:
        response = requests.post(
            f'{args.base_url}/ai/generate-print-layout-upload',
            files={'file': (image_path.name, fp, 'image/jpeg')},
            data=data,
            timeout=300,
        )
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
