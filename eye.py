#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IN YOUR EYES — живой глаз, целиком нарисованный ASCII-символами.

Установка:  python -m pip install numpy pygame
Запуск:    python ascii_eye.py

SPACE / ЛКМ — моргнуть; M — следить за мышью; C — цвет;
F / F11 — полный экран; +/- — размер символов; H — подсказка;
S — PNG в текущую папку; ESC — выход.

Никаких картинок, видео, моделей, загрузок или сетевых запросов при запуске.
Форма, кожа, сосуды, радужка и блики вычисляются математически.
Для экспорта GIF дополнительно нужен Pillow: python -m pip install pillow
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import numpy as np
    import pygame
except ImportError as exc:
    raise SystemExit(
        "Не хватает зависимости. Выполни:\n"
        "  python -m pip install numpy pygame\n"
        f"Подробности: {exc}"
    ) from exc


TAU = math.tau
PALETTES = ("amber", "frost", "emerald", "mono")
ALPHABET = " .,:;`'\"-~_=+<>/\\|!ilI()[]{}?1tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


def smooth(a, b, x):
    """Плавная ступенька: 0 слева от a, 1 справа от b."""
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def mix(a, b, weight):
    return a * (1.0 - weight[..., None]) + b * weight[..., None]


def noise2(x, y):
    """Неподвижный плавный шум для кожи; не случайный снег каждый кадр."""
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

    def h(a, b):
        v = np.sin(a * 127.1 + b * 311.7) * 43758.5453
        return v - np.floor(v)

    return ((1 - fy) * ((1 - fx) * h(ix, iy) + fx * h(ix + 1, iy))
            + fy * ((1 - fx) * h(ix, iy + 1) + fx * h(ix + 1, iy + 1)))


@dataclass
class EyeState:
    gx: float
    gy: float
    closed: float
    pupil: float
    tension: float


class Motion:
    """Саккады, фиксации, двойное моргание и плавная реакция зрачка."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.gaze = np.zeros(2, dtype=np.float64)
        self.start = self.gaze.copy()
        self.target = self.gaze.copy()
        self.move_at = 0.0
        self.move_duration = 0.16
        self.next_look = 1.3
        self.blink_at = -10.0
        self.blink_duration = 0.27
        self.next_blink = 2.7
        self.double_at = float("inf")
        self.pupil = 0.175
        self.pulse_at = -10.0

    def blink(self, t, startled=False, automatic=False):
        self.blink_at = t
        self.blink_duration = self.rng.uniform(0.23, 0.32)
        self.next_blink = t + self.rng.uniform(2.1, 4.7)
        self.double_at = (t + self.blink_duration + 0.12
                          if automatic and self.rng.random() < 0.22
                          else float("inf"))
        if startled:
            self.pulse_at = t

    def update(self, t, dt, mouse=None):
        if t >= self.double_at:
            self.blink(t)
        elif t >= self.next_blink:
            self.blink(t, automatic=True)

        if t >= self.next_look:
            self.start = self.gaze.copy()
            # Иногда пристальный взгляд точно на зрителя.
            if self.rng.random() < 0.33:
                self.target[:] = (0, -0.015)
                hold = self.rng.uniform(1.2, 2.1)
            else:
                self.target[:] = (self.rng.uniform(-0.36, 0.36),
                                  self.rng.uniform(-0.12, 0.13))
                hold = self.rng.uniform(0.55, 1.4)
            self.move_at = t
            self.move_duration = self.rng.uniform(0.11, 0.20)
            self.next_look = t + hold

        u = float(np.clip((t - self.move_at) / self.move_duration, 0, 1))
        # Minimum jerk: быстрое движение без механического линейного рывка.
        u = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
        wanted = self.start + (self.target - self.start) * u
        if mouse is not None:
            wanted = np.array((mouse[0] * 0.39, mouse[1] * 0.17))
        rate = 17.0 if mouse is not None else 46.0
        self.gaze += (wanted - self.gaze) * (1 - math.exp(-dt * rate))

        age = t - self.blink_at
        close_time = self.blink_duration * 0.28
        if 0 <= age < close_time:
            closed = float(smooth(0, close_time, age))
        elif close_time <= age < close_time + 0.025:
            closed = 1.0
        elif close_time + 0.025 <= age < self.blink_duration:
            closed = 1 - float(smooth(close_time + 0.025,
                                      self.blink_duration, age))
        else:
            closed = 0.0
        # При запуске глаз неспешно открывается.
        closed = max(closed, 1 - float(smooth(0.08, 0.95, t)))
        pulse = math.exp(-max(0, t - self.pulse_at) * 2.1)
        wanted_pupil = (0.176 + 0.018 * math.sin(t * 0.83)
                        + 0.009 * math.sin(t * 2.17) + pulse * 0.047)
        self.pupil += (wanted_pupil - self.pupil) * (1 - math.exp(-dt * 4.0))
        tremor_x = 0.0015 * math.sin(t * 31) + 0.002 * math.sin(t * 6.1)
        tremor_y = 0.0012 * math.sin(t * 23)
        return EyeState(float(self.gaze[0]) + tremor_x,
                        float(self.gaze[1]) + tremor_y,
                        closed, self.pupil,
                        0.5 + 0.5 * math.sin(t * 0.47))


class EyeShader:
    """Один отсчёт поверхности на одну символьную ячейку."""

    def __init__(self, cols, rows, cell_w, cell_h, size, seed):
        width, height = size
        scale = min(height * 0.5, width / 3.05)
        px = (np.arange(cols, dtype=np.float32) + 0.5) * cell_w
        py = (np.arange(rows, dtype=np.float32) + 0.5) * cell_h
        self.x, self.y = np.meshgrid((px - cols * cell_w / 2) / scale,
                                    (py - rows * cell_h * 0.52) / scale)
        x, y = self.x, self.y
        self.half_width = 1.40
        self.arch = np.clip(1 - (x / self.half_width) ** 2, 0, 1) ** 0.70
        self.corners = 0.030 + x * 0.035
        self.fine = noise2(x * 95 + seed, y * 95)
        self.grain = noise2(x * 37 + seed, y * 37)
        self.broad = noise2(x * 4 + seed, y * 4)
        self.vignette = np.exp(-0.26 * (x * x * 0.65 + (y + 0.03) ** 2))
        rng = np.random.default_rng(seed)
        self.angular = [rng.random(n).astype(np.float32)
                        for n in (89, 193, 487)]
        self.veins = self.make_veins()

    def make_veins(self):
        """Тонкие ветвящиеся сосуды от обоих уголков глаза."""
        x, y, w = self.x, self.y, self.half_width
        veins = np.zeros_like(x)
        for side in (-1, 1):
            u = (w - x * side) / 1.02
            visible = smooth(0.0, 0.13, u) * (1 - smooth(0.70, 1.0, u))
            for j, reach in enumerate((-0.38, -0.22, 0.20, 0.32)):
                path = (0.03 + side * 0.049 + reach * np.clip(u, 0, 1) ** 0.8
                        + 0.019 * np.sin(u * 13 + j * 2) * u)
                thickness = 0.0045 + 0.002 * (1 - np.clip(u, 0, 1))
                line = np.exp(-((y - path) / thickness) ** 2)
                branch = path + (j % 2 * 2 - 1) * 0.17 * np.maximum(u - 0.38, 0)
                line += (np.exp(-((y - branch) / 0.004) ** 2)
                         * smooth(0.34, 0.45, u) * 0.6)
                veins = np.maximum(veins, line * visible)
        return veins

    def angular_noise(self, angle, layer):
        values = self.angular[layer]
        p = angle * (len(values) / TAU)
        i = np.floor(p).astype(np.int32)
        f = p - i
        f = f * f * (3 - 2 * f)
        return values[i % len(values)] * (1 - f) + values[(i + 1) % len(values)] * f

    def shade(self, t, state, palette):
        x, y, arch = self.x, self.y, self.arch
        gx, gy = state.gx, state.gy
        breathe = math.sin(t * 1.15) * 0.006
        upper_height = 0.49 + breathe + gy * 0.12 + (state.tension - 0.5) * 0.025
        upper_rest = self.corners - upper_height * arch
        lower_rest = self.corners + (0.355 + breathe + gy * 0.10) * arch
        seam = self.corners + 0.052 * arch
        upper = upper_rest + (seam - upper_rest) * state.closed
        lower = lower_rest + (seam - lower_rest) * state.closed
        opening = (smooth(-0.009, 0.012, y - upper)
                   * smooth(-0.009, 0.012, lower - y)
                   * (1 - smooth(1.38, 1.41, np.abs(x))))

        # Кожа: тёплый объём, поры, тень над глазом и складки век.
        key_light = np.exp(-((x + 0.5) / 1.9) ** 2 - ((y - 0.30) / 1.1) ** 2)
        skin_light = (0.15 + 0.49 * key_light + 0.08 * self.broad
                      + 0.065 * (self.grain - 0.5) + 0.035 * (self.fine - 0.5))
        brow_y = -0.76 + 0.16 * (x / 1.40) ** 2
        brow = np.exp(-((y - brow_y) / 0.14) ** 2) * (1 - smooth(1.1, 1.5, np.abs(x)))
        socket = np.exp(-((y - upper + 0.075 * arch) / 0.16) ** 2)
        skin_light *= 1 - 0.57 * brow - 0.30 * socket
        skin = skin_light[..., None] * np.array((0.82, 0.51, 0.37), np.float32)
        crease = np.exp(-((y - upper + 0.115 * arch + 0.023) / 0.026) ** 2) * arch
        bag = np.exp(-((y - lower - 0.14 * arch - 0.036) / 0.036) ** 2) * arch
        skin *= (1 - 0.43 * crease - 0.21 * bag)[..., None]
        lower_roll = np.exp(-((y - lower - 0.065) / 0.060) ** 2) * arch
        skin += lower_roll[..., None] * np.array((0.075, 0.044, 0.031))

        # Склера — сферическая, с тенью века и красноватыми уголками.
        z = np.sqrt(np.clip(1 - ((x - gx * 0.25) / 1.65) ** 2
                            - ((y - gy * 0.18) / 0.89) ** 2, 0, 1))
        top_shadow = 0.58 + 0.42 * smooth(0.015, 0.23, y - upper)
        white_light = (0.22 + 0.70 * z) * top_shadow
        white_light *= 0.97 + 0.06 * self.grain
        sclera = white_light[..., None] * np.array((0.86, 0.91, 0.88))
        corner_red = smooth(0.76, 1.39, np.abs(x)) * 0.42
        sclera = mix(sclera, np.array((0.46, 0.20, 0.16)), corner_red)
        sclera = mix(sclera, np.array((0.39, 0.13, 0.105)), self.veins * 0.42)

        # Радужка движется с глазным яблоком. Перспектива сжимает её сбоку.
        dx = (x - gx) / (1 - abs(gx) * 0.17)
        dy = (y - gy) / (1 - abs(gy) * 0.12)
        radius = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        iris_r = 0.457
        pupil_r = state.pupil
        q = np.clip((radius - pupil_r) / (iris_r - pupil_r), 0, 1)
        bend = theta + 0.022 * np.sin(q * 12 + theta * 9)
        n1 = self.angular_noise(bend, 0)
        n2 = self.angular_noise(bend + 0.009 * np.sin(q * 23), 1)
        n3 = self.angular_noise(bend, 2)
        fibers = (0.40 * n1 + 0.30 * n2 + 0.14 * n3
                  + 0.10 * np.sin(theta * 173 + q * 16 + n1 * 7)
                  + 0.06 * np.sin(theta * 313 - q * 23))
        fibers = np.clip(fibers, 0, 1)
        crypts = smooth(0.65, 0.88, n1) * np.exp(-((q - 0.34) / 0.19) ** 2)
        collarette = np.exp(-((q - 0.30 - 0.075 * (n2 - 0.5)) / 0.075) ** 2)
        radial_light = 0.42 + 0.78 * fibers - 0.26 * crypts
        radial_light += 0.18 * collarette
        radial_light *= 1 - 0.73 * smooth(0.82, 1.0, radius / iris_r)
        radial_light *= 0.63 + 0.37 * smooth(-0.35, 0.23, dy)

        if palette == "frost":
            base, gold = (0.23, 0.65, 0.79), (0.12, 0.24, 0.27)
        elif palette == "emerald":
            base, gold = (0.40, 0.65, 0.28), (0.32, 0.17, 0.045)
        else:
            base, gold = (0.90, 0.43, 0.14), (0.23, 0.13, 0.028)
        iris = radial_light[..., None] * np.array(base)
        iris += (collarette * (0.3 + n2 * 0.7))[..., None] * np.array(gold)
        iris_mask = 1 - smooth(iris_r - 0.009, iris_r + 0.007, radius)
        globe = mix(sclera, iris, iris_mask)

        # Неровный зрачковый край, чёрный центр, тонкая пигментная кайма.
        jagged_pupil = pupil_r + 0.0024 * np.sin(theta * 61 + n2 * 3)
        pupil_mask = 1 - smooth(jagged_pupil - 0.003, jagged_pupil + 0.006, radius)
        globe = mix(globe, np.array((0.002, 0.003, 0.004)), pupil_mask)

        # Два окна в отражении роговицы; блики тоже состоят из символов.
        hx = dx + 0.115 + gx * 0.12
        hy = dy + 0.143 + gy * 0.13
        reflection = np.exp(-np.minimum((hx / 0.071) ** 8 + (hy / 0.080) ** 8, 80))
        panes = (0.30 + 0.70 * smooth(0.003, 0.013, np.abs(hx + 0.007)))
        panes *= 0.42 + 0.58 * smooth(0.002, 0.010, np.abs(hy + 0.006))
        reflection *= panes * 0.93
        small_glint = np.exp(-((dx - 0.16) / 0.026) ** 2 - ((dy - 0.17) / 0.022) ** 2) * 0.64
        reflection = np.clip(reflection + small_glint, 0, 0.98) * iris_mask
        globe = mix(globe, np.array((1.0, 0.98, 0.91)), reflection)

        # Тень ресниц, слёзная полоска и мясистый внутренний уголок.
        wet_rim = np.exp(-((y - lower + 0.012) / 0.014) ** 2) * arch
        globe += wet_rim[..., None] * np.array((0.21, 0.19, 0.14))
        duct = np.exp(-((x + 1.29) / 0.080) ** 2 - ((y + 0.012) / 0.064) ** 2)
        globe = mix(globe, np.array((0.49, 0.22, 0.19)), duct * 0.85)
        rgb = mix(skin, globe, opening)
        rim = np.exp(-((y - upper) / 0.018) ** 2) * arch
        rgb *= (1 - 0.76 * rim)[..., None]
        closed_line = np.exp(-((y - seam) / 0.018) ** 2) * arch * state.closed ** 4
        rgb *= (1 - 0.80 * closed_line)[..., None]

        # Верхние ресницы: корни следуют за веком при моргании.
        lash_id = np.rint((x + 1.27) / 0.105)
        root_x = lash_id * 0.105 - 1.27
        root_arch = np.clip(1 - (root_x / self.half_width) ** 2, 0, 1) ** 0.70
        root_corner = 0.03 + root_x * 0.035
        root_upper = root_corner - upper_height * root_arch
        root_seam = root_corner + 0.052 * root_arch
        root_y = root_upper + (root_seam - root_upper) * state.closed
        lash_length = 0.065 + 0.035 * (0.5 + 0.5 * np.sin(lash_id * 7.1))
        lash_distance = np.abs(x - root_x + (y - root_y) * np.sign(x) * 0.37)
        lashes = (np.exp(-(lash_distance / 0.009) ** 2)
                  * smooth(-0.006, 0.009, root_y - y)
                  * (1 - smooth(lash_length * 0.60, lash_length, root_y - y))
                  * (np.abs(root_x) < 1.30))
        rgb *= (1 - 0.77 * lashes)[..., None]
        rgb *= self.vignette[..., None]
        # Едва заметное дыхание света, без стробоскопа и резких вспышек.
        rgb *= 0.98 + 0.02 * math.sin(t * 0.63)
        if palette == "mono":
            gray = rgb @ np.array((0.2126, 0.7152, 0.0722))
            rgb = gray[..., None] * np.array((0.90, 0.98, 1.0))
        return np.clip(rgb, 0, 1).astype(np.float32)


class AsciiRenderer:
    """Атлас глифов + векторная раскраска: нет font.render() на каждый символ."""

    def __init__(self, size, font_size, seed):
        self.size = size
        font_path = pygame.font.match_font("consolas,menlo,dejavusansmono,liberationmono,couriernew")
        font = pygame.font.Font(font_path, font_size)
        self.cw = max(font.size(c)[0] for c in ALPHABET)
        self.ch = max(font.get_linesize(), font.size("M")[1])
        self.cols = max(1, size[0] // self.cw)
        self.rows = max(1, size[1] // self.ch)
        self.offset = ((size[0] - self.cols * self.cw) // 2,
                       (size[1] - self.rows * self.ch) // 2)
        tiles = []
        for char in ALPHABET:
            tile = pygame.Surface((self.cw, self.ch))
            glyph = font.render(char, True, (255, 255, 255))
            tile.blit(glyph, ((self.cw - glyph.get_width()) // 2, 0))
            tiles.append(pygame.surfarray.array3d(tile)[:, :, 0].T)
        tiles = np.array(tiles, dtype=np.uint8)
        # Сортируем по реальной площади штрихов выбранного системного шрифта.
        self.atlas = tiles[np.argsort(tiles.sum(axis=(1, 2)), kind="stable")]
        self.surface = pygame.Surface((self.cols * self.cw, self.rows * self.ch))
        shape = (self.rows, self.ch, self.cols, self.cw, 3)
        self.buffer16 = np.empty(shape, dtype=np.uint16)
        self.buffer8 = np.empty(shape, dtype=np.uint8)
        self.shader = EyeShader(self.cols, self.rows, self.cw, self.ch, size, seed)
        rng = np.random.default_rng(seed + 41)
        self.phase = rng.uniform(0, TAU, (self.rows, self.cols)).astype(np.float32)

    def draw(self, screen, t, state, palette):
        rgb = self.shader.shade(t, state, palette)
        luminance = rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
        shimmer = (np.sin(self.phase + t * 1.7) * 0.085
                   + np.sin(self.phase * 2.1 - t * 0.9) * 0.035)
        density = np.clip(luminance ** 0.46 + shimmer, 0, 1)
        indices = (density * (len(self.atlas) - 1)).astype(np.int32)
        indices[luminance < 0.009] = 0
        masks = self.atlas[indices].transpose(0, 2, 1, 3)
        colors = (np.clip(rgb ** 0.70 * 1.14, 0, 1) * 255).astype(np.uint16)
        # Переиспользуем буферы, раскладывая ячейки сразу в порядок пикселей.
        np.multiply(masks[..., None], (colors + 1)[:, None, :, None, :],
                    out=self.buffer16)
        np.right_shift(self.buffer16, 8, out=self.buffer16)
        np.copyto(self.buffer8, self.buffer16, casting="unsafe")
        pixels = self.buffer8.reshape(self.rows * self.ch, self.cols * self.cw, 3)
        pygame.surfarray.blit_array(self.surface, pixels.transpose(1, 0, 2))
        screen.fill((0, 0, 0))
        screen.blit(self.surface, self.offset)


def draw_help(screen, font):
    lines = ("IN YOUR EYES  /  ASCII",
             "SPACE / CLICK   blink", "M   mouse tracking", "C   iris color",
             "F / F11   fullscreen", "+ / -   character size",
             "S   save PNG", "H   hide help", "ESC   exit")
    panel = pygame.Surface((330, len(lines) * 24 + 26), pygame.SRCALPHA)
    panel.fill((5, 7, 9, 224))
    for i, line in enumerate(lines):
        panel.blit(font.render(line, True, (209, 183, 151)), (16, 13 + i * 24))
    screen.blit(panel, (20, 20))


def export(args, screen, renderer):
    """Фиксированный шаг времени: экспорт не зависит от скорости компьютера."""
    if args.gif:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SystemExit("Для GIF: python -m pip install pillow") from exc
    motion = Motion(args.seed)
    dt = 1 / args.export_fps
    end = args.time if args.still else args.seconds
    steps = max(1, math.ceil(end / dt))
    frames = []
    if args.gif:
        # Открытый глаз даёт общую палитру даже для тёмного первого кадра.
        renderer.draw(screen, 1.7, EyeState(0, 0, 0, 0.18, 0.5), args.palette)
        sample = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        palette = Image.fromarray(sample.copy()).quantize(colors=128)
    count = steps + 1 if args.still else steps
    for frame in range(count):
        t = min(frame * dt, end)
        state = motion.update(t, dt)
        if args.still and frame < steps:
            continue
        renderer.draw(screen, t, state, args.palette)
        if args.gif:
            array = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
            # Храним один индекс палитры на пиксель, а не три RGB-канала.
            frames.append(Image.fromarray(array.copy()).quantize(
                palette=palette, dither=Image.Dither.NONE))
        # Не копим SDL-события во время длинного экспорта.
        pygame.event.pump()
    path = Path(args.still or args.gif).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.still:
        pygame.image.save(screen, str(path))
    else:
        # Одна палитра на весь GIF предотвращает мерцание от переквантизации.
        # GIF хранит сотые доли секунды: распределяем округление по кадрам.
        boundaries = [round(min(i * dt, end) * 100) * 10
                      for i in range(len(frames) + 1)]
        durations = [max(10, b - a) for a, b in zip(boundaries, boundaries[1:])]
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=False, disposal=2)
    print(f"Saved: {path.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--font", type=int, default=14, help="Размер ASCII-символов")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--palette", choices=PALETTES, default="amber")
    parser.add_argument("--fullscreen", action="store_true")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--still", metavar="PNG", help="Сохранить кадр без открытия окна")
    outputs.add_argument("--gif", metavar="GIF", help="Сохранить GIF; нужен Pillow")
    parser.add_argument("--time", type=float, default=1.7, help="Время кадра для --still")
    parser.add_argument("--seconds", type=float, default=6.0, help="Длительность GIF")
    parser.add_argument("--export-fps", type=int, default=20)
    args = parser.parse_args(argv)
    if not (480 <= args.width <= 3840 and 320 <= args.height <= 2160):
        parser.error("Размер окна: от 480x320 до 3840x2160")
    if not 8 <= args.font <= 32 or not 1 <= args.fps <= 240:
        parser.error("--font: 8..32; --fps: 1..240")
    if not (1 <= args.export_fps <= 60 and 0 < args.seconds <= 30 and 0 <= args.time <= 120):
        parser.error("--export-fps: 1..60; --seconds: 0..30; --time: 0..120")
    if (args.gif or args.still) and args.fullscreen:
        parser.error("--fullscreen предназначен для интерактивного режима")
    gif_pixels = args.width * args.height * (math.ceil(args.seconds * args.export_fps) + 1)
    if args.gif and gif_pixels > 220_000_000:
        parser.error("Слишком большой GIF: уменьши --seconds, --export-fps или размер окна")
    if args.gif or args.still:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # Микшер не используется: не открываем аудиоустройства.
    pygame.display.init()
    pygame.font.init()
    window_size = (args.width, args.height)
    fullscreen = args.fullscreen
    flags = pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE
    screen = pygame.display.set_mode((0, 0) if fullscreen else window_size, flags)
    pygame.display.set_caption("IN YOUR EYES | ASCII | H: help")
    renderer = AsciiRenderer(screen.get_size(), args.font, args.seed)
    try:
        if args.still or args.gif:
            export(args, screen, renderer)
            return
        motion = Motion(args.seed)
        clock = pygame.time.Clock()
        help_font = pygame.font.Font(pygame.font.match_font("dejavusansmono,consolas"), 16)
        help_visible = False
        follow_mouse = False
        last_mouse = -10.0
        palette_index = PALETTES.index(args.palette)
        running = True
        t = 0.0
        last_caption = 0.0
        while running:
            # Ограничиваем шаг после перетаскивания окна / выхода из паузы ОС.
            dt = min(clock.tick(args.fps) / 1000.0, 0.05)
            t += dt
            rebuild = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE and not fullscreen:
                    window_size = (max(480, event.w), max(320, event.h))
                    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
                    rebuild = True
                elif event.type == pygame.MOUSEMOTION and any(event.rel):
                    last_mouse = t
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    motion.blink(t, startled=True)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        motion.blink(t)
                    elif event.key == pygame.K_h:
                        help_visible = not help_visible
                    elif event.key == pygame.K_m:
                        follow_mouse = not follow_mouse
                        last_mouse = -10.0
                    elif event.key == pygame.K_c:
                        palette_index = (palette_index + 1) % len(PALETTES)
                    elif event.key in (pygame.K_f, pygame.K_F11):
                        if not fullscreen:
                            window_size = screen.get_size()
                        fullscreen = not fullscreen
                        screen = pygame.display.set_mode(
                            (0, 0) if fullscreen else window_size,
                            pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
                        rebuild = True
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        args.font = min(32, args.font + 1)
                        rebuild = True
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        args.font = max(8, args.font - 1)
                        rebuild = True
                    elif event.key == pygame.K_s:
                        filename = f"eye_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}.png"
                        pygame.image.save(screen, filename)
                        print(f"Saved: {Path(filename).resolve()}")
            if not running:
                break
            if rebuild:
                renderer = AsciiRenderer(screen.get_size(), args.font, args.seed)
            mouse = None
            if pygame.mouse.get_focused() and (follow_mouse or t - last_mouse < 2.8):
                mx, my = pygame.mouse.get_pos()
                sw, sh = screen.get_size()
                mouse = (max(-1, min(1, mx / sw * 2 - 1)),
                         max(-1, min(1, my / sh * 2 - 1)))
            state = motion.update(t, dt, mouse)
            renderer.draw(screen, t, state, PALETTES[palette_index])
            if help_visible:
                draw_help(screen, help_font)
            pygame.display.flip()
            if t - last_caption > 1:
                mode = "MOUSE" if follow_mouse else "AUTO"
                pygame.display.set_caption(
                    f"IN YOUR EYES | {mode} | {clock.get_fps():.0f} FPS | H: help")
                last_caption = t
    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
