from __future__ import annotations

from dataclasses import dataclass, field

from utils.config import get_settings


@dataclass
class FacePostprocessResult:
    rawFaceCount: int
    validFaces: list[dict] = field(default_factory=list)
    primaryFaceBox: dict | None = None
    filteredOutReasons: list[dict] = field(default_factory=list)


class FacePostprocessService:
    def __init__(self):
        self.settings = get_settings()
        self.min_valid_face_width = self.settings.min_valid_face_width
        self.min_valid_face_height = self.settings.min_valid_face_height
        self.multi_face_min_area_ratio = self.settings.multi_face_min_area_ratio
        self.face_box_iou_threshold = self.settings.face_box_iou_threshold
        self.duplicate_overlap_ratio_threshold = max(
            self.face_box_iou_threshold,
            round(1 - self.face_box_iou_threshold, 2),
        )

    @staticmethod
    def _extract_value(face, key: str, fallback_key: str | None = None, default: int = 0) -> int:
        if isinstance(face, dict):
            if key in face:
                return int(face[key])
            if fallback_key and fallback_key in face:
                return int(face[fallback_key])
            return default

        names = getattr(getattr(face, 'dtype', None), 'names', None)
        if names:
            if key in names:
                return int(face[key])
            if fallback_key and fallback_key in names:
                return int(face[fallback_key])
        try:
            return int(face[key])
        except Exception:
            if fallback_key is not None:
                try:
                    return int(face[fallback_key])
                except Exception:
                    return default
            return default

    def normalize_face_boxes(self, faces) -> list[dict]:
        normalized: list[dict] = []
        for index, face in enumerate(faces):
            width = self._extract_value(face, 'width')
            height = self._extract_value(face, 'height')
            box = {
                'x': self._extract_value(face, 'x', fallback_key='c'),
                'y': self._extract_value(face, 'y', fallback_key='r'),
                'width': width,
                'height': height,
                'area': max(width, 0) * max(height, 0),
                'sourceIndex': index,
            }
            if box['width'] > 0 and box['height'] > 0:
                normalized.append(box)
        return normalized

    @staticmethod
    def _intersection_area(box_a: dict, box_b: dict) -> int:
        left = max(box_a['x'], box_b['x'])
        top = max(box_a['y'], box_b['y'])
        right = min(box_a['x'] + box_a['width'], box_b['x'] + box_b['width'])
        bottom = min(box_a['y'] + box_a['height'], box_b['y'] + box_b['height'])
        if right <= left or bottom <= top:
            return 0
        return (right - left) * (bottom - top)

    def _calc_iou(self, box_a: dict, box_b: dict) -> float:
        intersection = self._intersection_area(box_a, box_b)
        if intersection <= 0:
            return 0.0
        union = box_a['area'] + box_b['area'] - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _calc_overlap_over_smaller(self, box_a: dict, box_b: dict) -> float:
        intersection = self._intersection_area(box_a, box_b)
        smaller_area = min(box_a['area'], box_b['area'])
        if intersection <= 0 or smaller_area <= 0:
            return 0.0
        return intersection / smaller_area

    @staticmethod
    def _is_box_center_inside(target_box: dict, container_box: dict) -> bool:
        center_x = target_box['x'] + target_box['width'] / 2
        center_y = target_box['y'] + target_box['height'] / 2
        return (
            container_box['x'] <= center_x <= container_box['x'] + container_box['width']
            and container_box['y'] <= center_y <= container_box['y'] + container_box['height']
        )

    @staticmethod
    def _to_public_box(box: dict) -> dict:
        return {
            'x': int(box['x']),
            'y': int(box['y']),
            'width': int(box['width']),
            'height': int(box['height']),
        }

    def _is_duplicate_candidate(self, candidate: dict, kept: dict) -> bool:
        iou = self._calc_iou(candidate, kept)
        overlap_ratio = self._calc_overlap_over_smaller(candidate, kept)
        return (
            iou >= self.face_box_iou_threshold
            or overlap_ratio >= self.duplicate_overlap_ratio_threshold
            or self._is_box_center_inside(candidate, kept)
        )

    def face_box_postprocess(self, faces) -> FacePostprocessResult:
        raw_boxes = self.normalize_face_boxes(faces)
        filtered_out_reasons: list[dict] = []
        if not raw_boxes:
            return FacePostprocessResult(rawFaceCount=0)

        size_filtered: list[dict] = []
        for box in raw_boxes:
            if box['width'] < self.min_valid_face_width or box['height'] < self.min_valid_face_height:
                filtered_out_reasons.append(
                    {
                        'reason': 'too_small_absolute',
                        'box': self._to_public_box(box),
                    }
                )
                continue
            size_filtered.append(box)

        if not size_filtered:
            return FacePostprocessResult(
                rawFaceCount=len(raw_boxes),
                filteredOutReasons=filtered_out_reasons,
            )

        deduplicated: list[dict] = []
        for box in sorted(size_filtered, key=lambda item: item['area'], reverse=True):
            duplicate_of = next((kept for kept in deduplicated if self._is_duplicate_candidate(box, kept)), None)
            if duplicate_of is not None:
                filtered_out_reasons.append(
                    {
                        'reason': 'overlapped_duplicate',
                        'box': self._to_public_box(box),
                        'keptBox': self._to_public_box(duplicate_of),
                    }
                )
                continue
            deduplicated.append(box)

        if not deduplicated:
            return FacePostprocessResult(
                rawFaceCount=len(raw_boxes),
                filteredOutReasons=filtered_out_reasons,
            )

        primary_face = max(deduplicated, key=lambda item: item['area'])
        primary_area = max(primary_face['area'], 1)
        valid_faces: list[dict] = [primary_face]
        for box in deduplicated:
            if box is primary_face:
                continue
            area_ratio = box['area'] / primary_area
            if area_ratio < self.multi_face_min_area_ratio:
                filtered_out_reasons.append(
                    {
                        'reason': 'too_small_relative_to_primary',
                        'box': self._to_public_box(box),
                        'areaRatio': round(area_ratio, 3),
                    }
                )
                continue
            valid_faces.append(box)

        return FacePostprocessResult(
            rawFaceCount=len(raw_boxes),
            validFaces=[self._to_public_box(face) for face in valid_faces],
            primaryFaceBox=self._to_public_box(primary_face),
            filteredOutReasons=filtered_out_reasons,
        )
