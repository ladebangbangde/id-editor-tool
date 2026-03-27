from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from skimage.measure import perimeter
from skimage.color import rgb2hsv
from skimage.measure import find_contours
from skimage import measure, morphology

from app.services.photo_precheck_service import FAIL, PASS, WARNING
from app.core.config import get_settings


@dataclass
class OutputQualityResult:
    status: str
    reason_codes: list[str]
    warnings: list[str]
    primary_issue: str | None = None
    primary_message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    cloth_pollution_mask: Image.Image | None = None
    hair_gap_residue_mask: Image.Image | None = None


class OutputQualityService:
    FACE_POLLUTION_FAIL = 0.22
    FACE_POLLUTION_WARN = 0.14
    SKIN_RED_BLUE_CAST_FAIL = 40.0
    SKIN_RED_BLUE_CAST_WARN = 28.0
    SKIN_SAT_FAIL = 0.58
    SKIN_SAT_WARN = 0.48
    EDGE_NOISE_FAIL = 0.31
    EDGE_NOISE_WARN = 0.21
    FEATURE_POLLUTION_FAIL = 0.18
    FEATURE_POLLUTION_WARN = 0.10
    CLOTH_POLLUTION_FAIL = 0.27
    CLOTH_POLLUTION_WARN = 0.14
    HAIR_GAP_RESIDUE_FAIL = 0.010
    HAIR_GAP_RESIDUE_WARN = 0.004
    BORDER_RESIDUE_FAIL = 0.22
    BORDER_RESIDUE_WARN = 0.10

    ISSUE_PRIORITY = {
        'FACE_COLOR_POLLUTION': 100,
        'FACIAL_FEATURE_CORRUPTED': 95,
        'SKIN_TONE_ABNORMAL': 90,
        'FOREGROUND_EDGE_BROKEN': 85,
        'CLOTH_COLOR_POLLUTION': 84,
        'HAIR_GAP_BACKGROUND_RESIDUE': 83,
        'BORDER_BACKGROUND_RESIDUE': 82,
    }

    ISSUE_MESSAGES = {
        'FACE_COLOR_POLLUTION': '脸部检测到明显底色串色，请更换干净背景重试',
        'SKIN_TONE_ABNORMAL': '脸部肤色偏差较大，建议更换光线更均匀的照片',
        'FOREGROUND_EDGE_BROKEN': '人物边界质量异常，头发或肩部边缘存在破损风险',
        'FACIAL_FEATURE_CORRUPTED': '五官区域疑似受污染，建议重新处理或更换原图',
        'CLOTH_COLOR_POLLUTION': '衣领/肩部检测到明显底色侵入，建议切换更稳健前景保护模式',
        'HAIR_GAP_BACKGROUND_RESIDUE': '头发内部细缝存在漏底残留，建议使用更干净原图或重新抠图',
        'BORDER_BACKGROUND_RESIDUE': '画面边缘仍残留原始背景，背景替换不完整',
    }

    def __init__(self) -> None:
        self.settings = get_settings()

    def _cv2(self):
        try:
            import cv2

            return cv2
        except Exception:
            return None

    @staticmethod
    def _safe_face_box(face_box: dict[str, int] | None, width: int, height: int) -> dict[str, int] | None:
        if not face_box:
            return None
        x = max(0, min(width - 1, int(face_box.get('x', 0))))
        y = max(0, min(height - 1, int(face_box.get('y', 0))))
        w = max(1, min(width - x, int(face_box.get('width', 1))))
        h = max(1, min(height - y, int(face_box.get('height', 1))))
        return {'x': x, 'y': y, 'width': w, 'height': h}

    def _face_region_masks(self, face_box: dict[str, int], image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_shape
        x, y, fw, fh = face_box['x'], face_box['y'], face_box['width'], face_box['height']
        yy, xx = np.ogrid[:h, :w]
        cx = x + fw * 0.5
        cy = y + fh * 0.48
        rx = max(1.0, fw * 0.47)
        ry = max(1.0, fh * 0.55)
        face_core = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

        fx0 = int(max(0, x + fw * 0.18))
        fx1 = int(min(w, x + fw * 0.82))
        fy0 = int(max(0, y + fh * 0.22))
        fy1 = int(min(h, y + fh * 0.78))
        features = np.zeros((h, w), dtype=bool)
        features[fy0:fy1, fx0:fx1] = True
        return face_core, features

    def _cloth_region_mask(self, face_box: dict[str, int], image_shape: tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        x, y, fw, fh = face_box['x'], face_box['y'], face_box['width'], face_box['height']
        mask = np.zeros((h, w), dtype=bool)
        chest_top = int(min(h, y + fh * 0.92))
        chest_bottom = int(min(h, y + fh * 2.25))
        left = int(max(0, x - fw * 0.55))
        right = int(min(w, x + fw * 1.55))
        if chest_bottom > chest_top and right > left:
            mask[chest_top:chest_bottom, left:right] = True

        collar_top = int(min(h, y + fh * 0.70))
        collar_bottom = int(min(h, y + fh * 1.10))
        collar_left = int(max(0, x - fw * 0.05))
        collar_right = int(min(w, x + fw * 1.05))
        if collar_bottom > collar_top and collar_right > collar_left:
            mask[collar_top:collar_bottom, collar_left:collar_right] = True

        return mask

    def evaluate(
        self,
        source_image: Image.Image,
        output_image: Image.Image,
        foreground_rgba: Image.Image,
        face_box: dict[str, int] | None,
        background_color: str,
    ) -> OutputQualityResult:
        cv2 = self._cv2()
        src_rgb = np.asarray(source_image.convert('RGB'), dtype=np.uint8)
        out_rgb = np.asarray(output_image.convert('RGB'), dtype=np.uint8)
        alpha = np.asarray(foreground_rgba.convert('RGBA').getchannel('A'), dtype=np.uint8)

        h, w = out_rgb.shape[:2]
        safe_box = self._safe_face_box(face_box, w, h)

        bg_color = background_color.lower()
        reason_codes: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, float] = {}
        cloth_pollution_mask: Image.Image | None = None
        hair_gap_residue_mask: Image.Image | None = None

        if safe_box is not None:
            face_core, features_mask = self._face_region_masks(safe_box, (h, w))
            out_face = out_rgb[face_core]
            src_face = src_rgb[face_core] if src_rgb.shape[:2] == out_rgb.shape[:2] else out_face
            if out_face.size > 0:
                b = out_face[:, 2].astype(np.float32)
                g = out_face[:, 1].astype(np.float32)
                r = out_face[:, 0].astype(np.float32)
                blue_cast = float(np.mean(b - (r + g) * 0.5))
                red_cast = float(np.mean(r - (b + g) * 0.5))

                if cv2 is not None:
                    hsv = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2HSV)
                    sat = hsv[:, :, 1].astype(np.float32) / 255.0
                else:
                    sat = rgb2hsv(out_rgb.astype(np.float32) / 255.0)[:, :, 1].astype(np.float32)
                face_sat = float(np.mean(sat[face_core]))

                src_face_mean = np.mean(src_face.astype(np.float32), axis=0)
                out_face_mean = np.mean(out_face.astype(np.float32), axis=0)
                tone_shift = float(np.linalg.norm(out_face_mean - src_face_mean))

                pollution = red_cast if bg_color == 'red' else blue_cast if bg_color == 'blue' else max(red_cast, blue_cast)
                feature_region = out_rgb[features_mask]
                feature_pollution = 0.0
                if feature_region.size > 0:
                    fr = feature_region[:, 0].astype(np.float32)
                    fg = feature_region[:, 1].astype(np.float32)
                    fb = feature_region[:, 2].astype(np.float32)
                    target = fr - (fg + fb) * 0.5 if bg_color == 'red' else fb - (fr + fg) * 0.5 if bg_color == 'blue' else np.maximum(fr, fb) - fg
                    feature_pollution = float(np.mean(target > 20.0))

                metrics.update(
                    {
                        'face_color_pollution': float(pollution),
                        'face_saturation': face_sat,
                        'skin_tone_shift': tone_shift,
                        'feature_pollution_ratio': feature_pollution,
                    }
                )

                if pollution >= self.SKIN_RED_BLUE_CAST_FAIL:
                    reason_codes.append('FACE_COLOR_POLLUTION')
                elif pollution >= self.SKIN_RED_BLUE_CAST_WARN:
                    warnings.append('FACE_COLOR_POLLUTION')

                if tone_shift >= self.SKIN_RED_BLUE_CAST_FAIL or face_sat >= self.SKIN_SAT_FAIL:
                    reason_codes.append('SKIN_TONE_ABNORMAL')
                elif tone_shift >= self.SKIN_RED_BLUE_CAST_WARN or face_sat >= self.SKIN_SAT_WARN:
                    warnings.append('SKIN_TONE_ABNORMAL')

                if feature_pollution >= self.FEATURE_POLLUTION_FAIL:
                    reason_codes.append('FACIAL_FEATURE_CORRUPTED')
                elif feature_pollution >= self.FEATURE_POLLUTION_WARN:
                    warnings.append('FACIAL_FEATURE_CORRUPTED')

                if self.settings.enable_cloth_pollution_check and bg_color in {'red', 'blue'}:
                    cloth_mask = self._cloth_region_mask(safe_box, (h, w))
                    cloth_alpha_mask = cloth_mask & (alpha > 35)
                    if np.any(cloth_alpha_mask):
                        cloth_pixels = out_rgb[cloth_alpha_mask]
                        src_cloth_pixels = src_rgb[cloth_alpha_mask] if src_rgb.shape[:2] == out_rgb.shape[:2] else cloth_pixels
                        cr = cloth_pixels[:, 0].astype(np.float32)
                        cg = cloth_pixels[:, 1].astype(np.float32)
                        cb = cloth_pixels[:, 2].astype(np.float32)
                        if bg_color == 'red':
                            contamination_strength = cr - (cg + cb) * 0.5
                        else:
                            contamination_strength = cb - (cr + cg) * 0.5
                        contamination_binary = contamination_strength > 24.0
                        pollution_ratio = float(np.mean(contamination_binary))
                        src_mean = np.mean(src_cloth_pixels.astype(np.float32), axis=0)
                        out_mean = np.mean(cloth_pixels.astype(np.float32), axis=0)
                        cloth_color_shift = float(np.linalg.norm(out_mean - src_mean))
                        light_cloth_ratio = float(np.mean(np.mean(src_cloth_pixels.astype(np.float32), axis=1) > 150.0))
                        metrics['cloth_pollution_ratio'] = pollution_ratio
                        metrics['cloth_color_shift'] = cloth_color_shift
                        metrics['light_cloth_ratio'] = light_cloth_ratio

                        full_mask = np.zeros((h, w), dtype=np.uint8)
                        full_mask[cloth_alpha_mask] = (contamination_binary.astype(np.uint8) * 255)
                        cloth_pollution_mask = Image.fromarray(full_mask, mode='L')

                        high_risk = pollution_ratio >= self.CLOTH_POLLUTION_FAIL or (pollution_ratio >= self.CLOTH_POLLUTION_WARN and cloth_color_shift > 28.0 and light_cloth_ratio > 0.35)
                        medium_risk = pollution_ratio >= self.CLOTH_POLLUTION_WARN or (cloth_color_shift > 18.0 and light_cloth_ratio > 0.35)
                        if high_risk:
                            reason_codes.append('CLOTH_COLOR_POLLUTION')
                        elif medium_risk:
                            warnings.append('CLOTH_COLOR_POLLUTION')

                hair_top = int(max(0, safe_box['y'] - safe_box['height'] * 0.60))
                hair_bottom = int(min(h, safe_box['y'] + safe_box['height'] * 0.45))
                hair_left = int(max(0, safe_box['x'] - safe_box['width'] * 0.28))
                hair_right = int(min(w, safe_box['x'] + safe_box['width'] * 1.28))
                if hair_bottom > hair_top and hair_right > hair_left:
                    hair_region = np.zeros((h, w), dtype=bool)
                    hair_region[hair_top:hair_bottom, hair_left:hair_right] = True
                    hair_rgb = out_rgb.astype(np.float32)
                    if cv2 is not None:
                        hsv_full = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                        sat = hsv_full[:, :, 1] / 255.0
                    else:
                        sat = rgb2hsv(out_rgb.astype(np.float32) / 255.0)[:, :, 1].astype(np.float32)

                    bright_low_sat = (np.mean(hair_rgb, axis=2) > 214.0) & (sat < 0.16)
                    candidate = hair_region & bright_low_sat
                    labels = measure.label(candidate, connectivity=2)
                    residue_mask = np.zeros((h, w), dtype=np.uint8)
                    for region in measure.regionprops(labels):
                        if region.area < 2 or region.area > 140:
                            continue
                        comp = labels == region.label
                        ring = morphology.binary_dilation(comp, morphology.disk(2)) & (~comp)
                        if not np.any(ring):
                            continue
                        ring_fg_ratio = float(np.mean(alpha[ring] > 85))
                        ring_dark_ratio = float(np.mean(np.mean(hair_rgb[ring], axis=1) < 165.0))
                        if ring_fg_ratio > 0.42 and ring_dark_ratio > 0.28:
                            residue_mask[comp] = 255

                    if np.any(hair_region):
                        hair_gap_ratio = float(np.mean(residue_mask[hair_region] > 0))
                        metrics['hair_gap_residue_ratio'] = hair_gap_ratio
                        if np.any(residue_mask):
                            hair_gap_residue_mask = Image.fromarray(residue_mask, mode='L')
                        if hair_gap_ratio >= self.HAIR_GAP_RESIDUE_FAIL:
                            reason_codes.append('HAIR_GAP_BACKGROUND_RESIDUE')
                        elif hair_gap_ratio >= self.HAIR_GAP_RESIDUE_WARN:
                            warnings.append('HAIR_GAP_BACKGROUND_RESIDUE')

                if bg_color in {'red', 'blue'}:
                    strip = max(4, min(h, w) // 20)
                    border_mask = np.zeros((h, w), dtype=bool)
                    border_mask[:strip, :] = True
                    border_mask[-strip:, :] = True
                    border_mask[:, :strip] = True
                    border_mask[:, -strip:] = True

                    border_pixels = out_rgb[border_mask]
                    if border_pixels.size > 0:
                        if bg_color == 'blue':
                            target = np.array([67.0, 142.0, 219.0], dtype=np.float32)
                        else:
                            target = np.array([230.0, 50.0, 55.0], dtype=np.float32)
                        pixel_f = border_pixels.astype(np.float32)
                        bright = np.mean(pixel_f, axis=1) > 205.0
                        low_sat = (np.max(pixel_f, axis=1) - np.min(pixel_f, axis=1)) < 25.0
                        target_dist = np.linalg.norm(pixel_f - target[None, :], axis=1)
                        residue_ratio = float(np.mean(bright & low_sat & (target_dist > 55.0)))
                        metrics['border_background_residue_ratio'] = residue_ratio
                        if residue_ratio >= self.BORDER_RESIDUE_FAIL:
                            reason_codes.append('BORDER_BACKGROUND_RESIDUE')
                        elif residue_ratio >= self.BORDER_RESIDUE_WARN:
                            warnings.append('BORDER_BACKGROUND_RESIDUE')

        edge_band = (alpha > 8) & (alpha < 248)
        edge_noise = 0.0
        if np.any(edge_band):
            region = edge_band.astype(bool)
            if cv2 is not None:
                edge_u8 = region.astype(np.uint8)
                contours, _ = cv2.findContours(edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour_len = sum(float(cv2.arcLength(cnt, closed=True)) for cnt in contours)
            else:
                contour_len = 0.0
                for contour in find_contours(region.astype(np.float32), 0.5):
                    if contour.shape[0] > 1:
                        diffs = np.diff(contour, axis=0)
                        contour_len += float(np.sum(np.linalg.norm(diffs, axis=1)))
            area = float(np.count_nonzero(region))
            perimeter_len = float(perimeter(region)) if area > 0 else 0.0
            edge_noise = contour_len / max(perimeter_len, 1.0)
            metrics['edge_noise_score'] = edge_noise

            if edge_noise >= self.EDGE_NOISE_FAIL:
                reason_codes.append('FOREGROUND_EDGE_BROKEN')
            elif edge_noise >= self.EDGE_NOISE_WARN:
                warnings.append('FOREGROUND_EDGE_BROKEN')

        reason_codes = sorted(set(reason_codes), key=lambda code: -self.ISSUE_PRIORITY.get(code, 0))
        warnings = sorted(set(warnings), key=lambda code: -self.ISSUE_PRIORITY.get(code, 0))

        if reason_codes:
            primary = reason_codes[0]
            return OutputQualityResult(
                status=FAIL,
                reason_codes=reason_codes,
                warnings=warnings,
                primary_issue=primary,
                primary_message=self.ISSUE_MESSAGES.get(primary),
                metrics=metrics,
                cloth_pollution_mask=cloth_pollution_mask,
                hair_gap_residue_mask=hair_gap_residue_mask,
            )

        if warnings:
            primary = warnings[0]
            return OutputQualityResult(
                status=WARNING,
                reason_codes=[],
                warnings=warnings,
                primary_issue=primary,
                primary_message=self.ISSUE_MESSAGES.get(primary),
                metrics=metrics,
                cloth_pollution_mask=cloth_pollution_mask,
                hair_gap_residue_mask=hair_gap_residue_mask,
            )

        return OutputQualityResult(
            status=PASS,
            reason_codes=[],
            warnings=[],
            primary_issue=None,
            primary_message='输出成片质量正常',
            metrics=metrics,
            cloth_pollution_mask=cloth_pollution_mask,
            hair_gap_residue_mask=hair_gap_residue_mask,
        )
