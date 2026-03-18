from pathlib import Path

from PIL import Image
from skimage import data


if __name__ == '__main__':
    target = Path('inputs/sample_astronaut.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data.astronaut()).save(target)
    print(target.resolve())
