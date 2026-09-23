from __future__ import annotations

import ctypes
import io
import re
import time
from dataclasses import dataclass
from typing import Any

from ..errors import CapabilityUnavailable
from ..vision import WindowImage


@dataclass
class ScreenController:
    """Physical mouse/keyboard driver with semantic window and control location.

    UI Automation may locate a rectangle, but all user input is emitted by
    PyAutoGUI so headed demos show the same interaction a person would perform.
    """

    pause_s: float = 0.55
    move_s: float = 0.45

    @staticmethod
    def _modules():
        try:
            import pyautogui
            from pywinauto import Desktop
        except ImportError:
            raise CapabilityUnavailable(
                "install cua-jev[windows] for physical screen GUI execution"
            ) from None
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.12
        return pyautogui, Desktop

    def focus(self, title_re: str, *, maximize: bool = True, timeout_s: float = 15):
        _, desktop = self._modules()
        deadline = time.time() + timeout_s
        matches = []
        while time.time() < deadline and not matches:
            matches = desktop(backend="uia").windows(title_re=title_re, visible_only=True)
            if not matches:
                time.sleep(0.2)
        if not matches:
            raise RuntimeError(f"visible window not found: {title_re}")
        window = desktop(backend="uia").window(handle=matches[-1].handle)
        window.set_focus()
        if maximize:
            try:
                window.maximize()
            except Exception:
                pass
        time.sleep(self.pause_s)
        return window

    def focus_handle(self, handle: int, *, maximize: bool = True):
        _, desktop = self._modules()
        window = desktop(backend="uia").window(handle=int(handle))
        window.wait("exists enabled visible ready", timeout=15)
        window.set_focus()
        if maximize:
            try:
                window.maximize()
            except Exception:
                pass
        time.sleep(self.pause_s)
        return window

    def click_point(self, x: float, y: float) -> None:
        pyautogui, _ = self._modules()
        pyautogui.moveTo(round(x), round(y), duration=self.move_s)
        pyautogui.click()
        time.sleep(self.pause_s)

    def capture_region(self, region: tuple[int, int, int, int]) -> WindowImage:
        """Capture only the selected window; image bytes stay in memory."""
        left, top, width, height = region
        if width < 40 or height < 40 or width * height > 16_000_000:
            raise ValueError("window screenshot region is outside safe size bounds")
        pyautogui, _ = self._modules()
        image = pyautogui.screenshot(region=(left, top, width, height)).convert("RGB")
        sample = image.convert("L").resize((64, 64)).tobytes()
        image.thumbnail((1280, 1280))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72)
        return WindowImage(buffer.getvalue(), sample, region)

    @staticmethod
    def ensure_foreground(handle: int) -> None:
        """Fail closed if another window has focus before a visual capture/click."""
        foreground = int(ctypes.windll.user32.GetForegroundWindow())
        if foreground != int(handle):
            raise RuntimeError("selected window is not foreground")

    def move_point(self, x: float, y: float) -> None:
        pyautogui, _ = self._modules()
        pyautogui.moveTo(round(x), round(y), duration=self.move_s)
        time.sleep(self.pause_s)

    def click_control(
        self,
        title_re: str,
        *,
        control_title: str,
        control_type: str,
    ) -> dict[str, Any]:
        window = self.focus(title_re)
        names = {re.escape(control_title), re.escape(control_title.rsplit(".", 1)[0])}
        control = window.child_window(
            title_re=rf"^(?:{'|'.join(sorted(names))})$",
            control_type=control_type,
        )
        control.wait("exists enabled visible ready", timeout=10)
        rectangle = control.rectangle()
        self.click_point(rectangle.mid_point().x, rectangle.mid_point().y)
        return {
            "window": window.window_text(),
            "control": control_title,
            "point": [rectangle.mid_point().x, rectangle.mid_point().y],
        }

    def click_control_handle(
        self,
        handle: int,
        *,
        control_title: str,
        control_type: str,
    ) -> dict[str, Any]:
        window = self.focus_handle(handle)
        names = {re.escape(control_title), re.escape(control_title.rsplit(".", 1)[0])}
        control = window.child_window(
            title_re=rf"^(?:{'|'.join(sorted(names))})$",
            control_type=control_type,
        )
        control.wait("exists enabled visible ready", timeout=10)
        rectangle = control.rectangle()
        self.click_point(rectangle.mid_point().x, rectangle.mid_point().y)
        return {
            "window": window.window_text(),
            "control": control_title,
            "point": [rectangle.mid_point().x, rectangle.mid_point().y],
        }

    def hotkey(self, *keys: str) -> None:
        pyautogui, _ = self._modules()
        pyautogui.hotkey(*keys)
        time.sleep(self.pause_s)

    def press(self, key: str) -> None:
        pyautogui, _ = self._modules()
        pyautogui.press(key)
        time.sleep(self.pause_s)

    def scroll(self, clicks: int) -> None:
        pyautogui, _ = self._modules()
        pyautogui.scroll(clicks)
        time.sleep(self.pause_s)

    def write(self, text: str, *, interval: float = 0.035) -> None:
        pyautogui, _ = self._modules()
        pyautogui.write(text, interval=interval)
        time.sleep(self.pause_s)

    def paste_text(self, text: str) -> None:
        try:
            import win32clipboard
        except ImportError:
            raise CapabilityUnavailable("install cua-jev[windows] for clipboard input") from None
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text)
        finally:
            win32clipboard.CloseClipboard()
        self.hotkey("ctrl", "v")
