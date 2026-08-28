import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # Встроенный в MoviePy SlideIn в текущей цепочке обработки нестабилен на
    # полноэкранных материалах: переход формально применяется, а на картинке
    # практически ничего не меняется. Заменяем его явной чёрной подложкой и
    # анимацией сдвига — так переход заметен, а поведение предсказуемо.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_start = max(clip.duration - t, 0)

    # SlideOut тоже заменён явным сдвигом, чтобы конец фрагмента стабильно уезжал за кадр.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(t, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Сохраняем исходную амплитуду масштабирования в 20%, чтобы даже трёхсекундный ролик давал заметное движение в духе Кена Бёрнса.
# Стабильность масштабирования обеспечивает субпиксельная выборка по центру ниже, а не ослабление эффекта ради маскировки мерцания кодека исходника.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Стабильное масштабирование без чёрных полей за счёт субпиксельного кадрирования по центру.

    Приводить ширину и высоту кадрирования к целым числам заранее нельзя: при
    непрерывном изменении масштаба целочисленные границы прыгают с разным шагом,
    а на переходе между чётным и нечётным размером меняется фаза полупиксельной
    выборки — в итоге картинка дрожит. Преобразование EXTENT из Pillow принимает
    дробные границы и выполняет субпиксельную выборку на холсте фиксированного
    размера; левая и правая, верхняя и нижняя границы всегда симметричны
    относительно одного дробного центра, поэтому подход годится для медленного
    масштабирования на протяжении всего видео.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # При масштабе 1 возвращаем исходный кадр: бессмысленная передискретизация слегка размыла бы первый кадр.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # При непрерывном масштабировании важнее согласованность соседних кадров.
        # BICUBIC и LANCZOS дают более резкий отдельный кадр, но на высокочастотных
        # текстурах, пересекающих сетку выборки, легко появляются звон и мерцание
        # яркости. BILINEAR мягче и небольшой потерей резкости покупает более стабильную картинку в движении.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Плавно увеличивает изображение от исходного до 1.2× на протяжении всего фрагмента."""
    # Параметр t пока сохранён ради единой сигнатуры с другими функциями переходов.
    # Масштабирование должно покрывать весь фрагмент: иначе после короткого зума картинка резко замирает, что плохо для статичных и малоподвижных материалов.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """Плавно уменьшает изображение с 1.2× до исходного на протяжении всего фрагмента."""
    # Как и в zoomin_transition, t нужен только для совместимости с общим интерфейсом вызова переходов.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)
