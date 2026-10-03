"""Host/Join desktop window. Logic lives in ``controller``; this is the view."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional

from slashcompute.launcher.controller import Launcher, LauncherError, LauncherSettings

# Evening desk: charcoal board, warm paper type, one copper start control.
# Native macOS Tk buttons ignore fill, so actions are painted Labels.
BG = "#161412"
SURFACE = "#211e1b"
LINE = "#3a342e"
INK = "#f3ece3"
MUTED = "#b5a99a"
COPPER = "#c46a2b"
COPPER_INK = "#fff7ee"
DANGER = "#c44b32"
OK = "#3d8a5a"
PAD = 28
WIDTH = 360


class PaintButton:
    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        font: tkfont.Font,
        fg: str,
        bg: str,
        command: Callable[[], None],
        pady: int = 10,
        padx: int = 14,
        hover: Optional[str] = None,
    ) -> None:
        self._command = command
        self._enabled = True
        self._fg, self._bg = fg, bg
        self._hover = hover or bg
        self.label = tk.Label(
            parent, text=text, font=font, fg=fg, bg=bg, pady=pady, padx=padx,
            cursor="hand2",
        )
        self.label.bind("<Button-1>", self._click)
        self.label.bind("<Enter>", lambda _e: self._tint(True))
        self.label.bind("<Leave>", lambda _e: self._tint(False))

    def _tint(self, on: bool) -> None:
        if not self._enabled:
            return
        self.label.configure(bg=self._hover if on else self._bg)

    def _click(self, _e=None) -> None:
        if self._enabled:
            self._command()

    def configure(self, text: Optional[str] = None, state: Optional[str] = None, **kw) -> None:
        if text is not None:
            self.label.configure(text=text)
        if "fg" in kw:
            self._fg = kw["fg"]
            self.label.configure(fg=kw["fg"])
        if "bg" in kw:
            self._bg = kw["bg"]
            self.label.configure(bg=kw["bg"])
        if state is not None:
            self._enabled = state != tk.DISABLED
            if self._enabled:
                self.label.configure(fg=self._fg, bg=self._bg, cursor="hand2")
            else:
                self.label.configure(fg=MUTED, bg=SURFACE, cursor="arrow")

    def pack(self, **kw) -> None:
        self.label.pack(**kw)

    def cget(self, key: str):
        return self.label.cget(key)


class GpuSlider:
    def __init__(self, parent: tk.Widget, variable: tk.IntVar, on_change: Callable[[], None]) -> None:
        self.var = variable
        self.on_change = on_change
        self.cv = tk.Canvas(parent, height=26, bg=BG, highlightthickness=0, cursor="hand2")
        self.cv.bind("<Button-1>", self._drag)
        self.cv.bind("<B1-Motion>", self._drag)
        self.cv.bind("<Configure>", lambda _e: self.draw())

    def pack(self, **kw) -> None:
        self.cv.pack(**kw)

    def _drag(self, e) -> None:
        w = max(1, self.cv.winfo_width() - 12)
        pct = max(1, min(100, int(round((e.x - 6) / w * 99) + 1)))
        self.var.set(pct)
        self.draw()
        self.on_change()

    def draw(self) -> None:
        self.cv.delete("all")
        w = self.cv.winfo_width()
        if w < 8:
            return
        y = 13
        self.cv.create_line(6, y, w - 6, y, fill=LINE, width=4)
        x = 6 + (self.var.get() - 1) / 99 * (w - 12)
        self.cv.create_line(6, y, x, y, fill=COPPER, width=4)
        self.cv.create_oval(x - 7, y - 7, x + 7, y + 7, fill=INK, outline=COPPER, width=2)


class LauncherWindow:
    def __init__(self, launcher: Optional[Launcher] = None) -> None:
        self.launcher = launcher or Launcher()
        self.settings = self.launcher.load_settings()
        self.root = tk.Tk()
        self.root.title("/compute")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._mode = tk.StringVar(value=self.settings.mode)
        self._url = tk.StringVar(value=self.settings.url)
        self._gpu = tk.IntVar(value=self.settings.gpu_percent)
        self._contribute = tk.BooleanVar(value=self.settings.contribute)
        self._error = ""
        self._searching = False
        self._busy = False

        self._fonts()
        self._build()
        self._show_mode()
        self._refresh()
        self.root.after(1000, self._tick)

    def _fonts(self) -> None:
        family = "SF Pro Text"
        available = set(tkfont.families(self.root))
        if family not in available:
            family = ".AppleSystemUIFont" if ".AppleSystemUIFont" in available else "Helvetica Neue"
        display_family = "SF Pro Display" if "SF Pro Display" in available else family
        self.font_display = tkfont.Font(family=display_family, size=26, weight="bold")
        self.font_title = tkfont.Font(family=family, size=13, weight="bold")
        self.font_body = tkfont.Font(family=family, size=13)
        self.font_small = tkfont.Font(family=family, size=11)
        self.font_mono = tkfont.Font(family="Menlo", size=13)

    def _build(self) -> None:
        shell = tk.Frame(self.root, bg=BG, padx=PAD, pady=PAD, width=WIDTH + PAD * 2)
        shell.pack(fill=tk.BOTH, expand=True)

        tk.Label(shell, text="/compute", font=self.font_display, fg=INK, bg=BG,
                 anchor="w").pack(fill=tk.X)
        tk.Label(
            shell, text="Start a pool on this Mac, or join one already running.",
            font=self.font_body, fg=MUTED, bg=BG, anchor="w", wraplength=WIDTH, justify="left",
        ).pack(fill=tk.X, pady=(8, 20))

        self._mode_bar = tk.Frame(shell, bg=SURFACE, highlightbackground=LINE,
                                  highlightthickness=1)
        self._mode_bar.pack(fill=tk.X)
        self._host_btn = PaintButton(
            self._mode_bar, "Host pool", self.font_title, MUTED, SURFACE,
            lambda: self._set_mode("host"),
        )
        self._join_btn = PaintButton(
            self._mode_bar, "Join pool", self.font_title, MUTED, SURFACE,
            lambda: self._set_mode("join"),
        )
        self._host_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._join_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._host_panel = tk.Frame(shell, bg=BG)
        tk.Label(self._host_panel, text="This Mac’s address", font=self.font_small,
                 fg=MUTED, bg=BG, anchor="w").pack(fill=tk.X, pady=(18, 6))
        ip_row = tk.Frame(self._host_panel, bg=SURFACE, highlightbackground=LINE,
                          highlightthickness=1)
        ip_row.pack(fill=tk.X)
        self._ip = tk.Label(ip_row, text=self.launcher.snapshot().lan_ip, font=self.font_mono,
                            fg=INK, bg=SURFACE, anchor="w", padx=12, pady=10)
        self._ip.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._copy_btn = PaintButton(
            ip_row, "Copy", self.font_small, INK, SURFACE, self._copy_ip,
            pady=8, padx=12, hover=LINE,
        )
        self._copy_btn.pack(side=tk.RIGHT)
        self._contrib = tk.Checkbutton(
            self._host_panel, text="Also contribute this Mac", variable=self._contribute,
            font=self.font_body, fg=INK, bg=BG, activebackground=BG, activeforeground=INK,
            selectcolor=SURFACE, highlightthickness=0, command=self._persist,
            anchor="w", pady=10,
        )
        self._contrib.pack(fill=tk.X)

        self._join_panel = tk.Frame(shell, bg=BG)
        tk.Label(self._join_panel, text="Coordinator URL", font=self.font_small,
                 fg=MUTED, bg=BG, anchor="w").pack(fill=tk.X, pady=(18, 6))
        url_row = tk.Frame(self._join_panel, bg=SURFACE, highlightbackground=LINE,
                           highlightthickness=1)
        url_row.pack(fill=tk.X)
        self._url_entry = tk.Entry(
            url_row, textvariable=self._url, font=self.font_mono, fg=INK, bg=SURFACE,
            insertbackground=INK, relief=tk.FLAT, highlightthickness=0,
        )
        self._url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12, pady=10)
        self._url_entry.bind("<FocusOut>", lambda _e: self._persist())
        self._url_entry.bind("<Return>", lambda _e: self._on_start())
        find_row = tk.Frame(self._join_panel, bg=BG)
        find_row.pack(fill=tk.X, pady=(10, 0))
        find_wrap = tk.Frame(find_row, bg=LINE, padx=1, pady=1)
        find_wrap.pack(side=tk.LEFT)
        self._find_btn = PaintButton(
            find_wrap, "Find on LAN", self.font_small, INK, SURFACE, self._find,
            pady=7, padx=12, hover=LINE,
        )
        self._find_btn.pack()

        gpu = tk.Frame(shell, bg=BG)
        gpu.pack(fill=tk.X, pady=(16, 0))
        head = tk.Frame(gpu, bg=BG)
        head.pack(fill=tk.X)
        tk.Label(head, text="GPU time", font=self.font_title, fg=INK, bg=BG,
                 anchor="w").pack(side=tk.LEFT)
        self._gpu_label = tk.Label(head, text=f"{self._gpu.get()}%", font=self.font_mono,
                                   fg=MUTED, bg=BG, anchor="e")
        self._gpu_label.pack(side=tk.RIGHT)
        self._slider = GpuSlider(gpu, self._gpu, self._on_gpu_change)
        self._slider.pack(fill=tk.X, pady=(6, 0))

        btns = tk.Frame(shell, bg=BG)
        btns.pack(fill=tk.X, pady=(20, 0))
        self._start = PaintButton(
            btns, "Start", self.font_title, COPPER_INK, COPPER, self._on_start,
            pady=12, hover="#a85720",
        )
        self._start.pack(fill=tk.X)
        self._stop = PaintButton(
            btns, "Stop", self.font_body, INK, SURFACE, self._on_stop,
            pady=10, hover=LINE,
        )
        self._stop.label.configure(highlightbackground=LINE, highlightthickness=1)
        self._stop.pack(fill=tk.X, pady=(8, 0))

        self._status = tk.Frame(shell, bg=SURFACE, highlightbackground=LINE,
                                highlightthickness=1)
        self._status.pack(fill=tk.X, pady=(20, 0))
        self._coord_row = self._status_row(self._status, "Coordinator")
        self._agent_row = self._status_row(self._status, "Agent")
        self._nodes_row = self._status_row(self._status, "Macs in pool")

        self._err = tk.Label(shell, text="", font=self.font_small, fg=DANGER, bg=BG,
                             anchor="w", wraplength=WIDTH, justify="left")

        self._hint = tk.Label(
            shell, text="Allow incoming TCP 8765 on the host and 9700 on each agent.",
            font=self.font_small, fg=MUTED, bg=BG, anchor="w", wraplength=WIDTH,
            justify="left",
        )
        self._hint.pack(fill=tk.X, pady=(12, 0))

    def _status_row(self, parent: tk.Widget, label: str) -> dict:
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill=tk.X, padx=12, pady=7)
        dot = tk.Canvas(row, width=10, height=10, bg=SURFACE, highlightthickness=0)
        dot.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text=label, font=self.font_body, fg=MUTED, bg=SURFACE,
                 anchor="w").pack(side=tk.LEFT)
        value = tk.Label(row, text="—", font=self.font_body, fg=INK, bg=SURFACE, anchor="e")
        value.pack(side=tk.RIGHT)
        return {"dot": dot, "value": value}

    def _set_mode(self, mode: str) -> None:
        self._mode.set(mode)
        self._persist()
        self._show_mode()
        self._refresh()

    def _show_mode(self) -> None:
        host = self._mode.get() == "host"
        self._paint_seg(self._host_btn, host)
        self._paint_seg(self._join_btn, not host)
        if host:
            self._join_panel.pack_forget()
            self._host_panel.pack(fill=tk.X, after=self._mode_bar)
            self._hint.configure(
                text="Allow incoming TCP 8765 on this Mac and 9700 on each agent.",
            )
        else:
            self._host_panel.pack_forget()
            self._join_panel.pack(fill=tk.X, after=self._mode_bar)
            self._hint.configure(text="Allow incoming TCP 9700 on this Mac.")

    def _paint_seg(self, btn: PaintButton, on: bool) -> None:
        if on:
            btn.configure(fg=BG, bg=INK)
            btn._hover = INK
        else:
            btn.configure(fg=MUTED, bg=SURFACE)
            btn._hover = LINE

    def _current(self) -> LauncherSettings:
        return LauncherSettings(
            mode=self._mode.get(), url=self._url.get(),
            gpu_percent=int(self._gpu.get()), contribute=bool(self._contribute.get()),
        )

    def _persist(self) -> None:
        self.launcher.save_settings(self._current())

    def _on_gpu_change(self) -> None:
        self._gpu_label.configure(text=f"{int(self._gpu.get())}%")
        self._persist()

    def _copy_ip(self) -> None:
        ip = self._ip.cget("text")
        self.root.clipboard_clear()
        self.root.clipboard_append(ip)
        self._copy_btn.configure(text="Copied")
        self.root.after(1400, lambda: self._copy_btn.configure(text="Copy"))

    def _find(self) -> None:
        if self._searching:
            return
        self._searching = True
        self._find_btn.configure(text="Searching…", state=tk.DISABLED)

        def work() -> None:
            err = ""
            found = None
            try:
                found = self.launcher.find_on_lan()
            except Exception as e:
                err = str(e)
            self.root.after(0, lambda: self._find_done(found, err))

        threading.Thread(target=work, daemon=True).start()

    def _find_done(self, found: Optional[str], err: str) -> None:
        self._searching = False
        self._find_btn.configure(text="Find on LAN", state=tk.NORMAL)
        if found:
            self._url.set(found)
            self._error = ""
            self._persist()
        else:
            self._error = err or "No coordinator found on the LAN. Type the host IP."
        self._refresh()

    def _on_start(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._start.configure(text="Starting…", state=tk.DISABLED)
        self.root.update_idletasks()
        try:
            self.launcher.start(self._current())
            self._error = self.launcher.last_error
        except LauncherError as e:
            self._error = str(e)
        finally:
            self._busy = False
            self._start.configure(text="Start", state=tk.NORMAL)
            self._refresh()

    def _on_stop(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            self.launcher.stop()
            self._error = self.launcher.last_error
        except LauncherError as e:
            self._error = str(e)
        finally:
            self._busy = False
            self._refresh()

    def _tick(self) -> None:
        self._refresh()
        self.root.after(1000, self._tick)

    def _refresh(self) -> None:
        snap = self.launcher.snapshot(self._current())
        self._ip.configure(text=snap.lan_ip)
        err = self._error or snap.last_error
        self._err.configure(text=err)
        if err:
            if not self._err.winfo_ismapped():
                self._err.pack(fill=tk.X, pady=(8, 0), before=self._hint)
        else:
            self._err.pack_forget()

        if snap.coordinator_up:
            extra = f"{snap.nodes} connected" if snap.nodes else "up"
            self._set_row(self._coord_row, True, extra)
        else:
            self._set_row(self._coord_row, False, "offline")

        if snap.agent_running:
            label = snap.agent_status or "running"
            if snap.agent_job_id:
                label = f"{label} · job {snap.agent_job_id[:8]}"
            self._set_row(self._agent_row, True, label)
        else:
            self._set_row(self._agent_row, False, "stopped")

        if snap.coordinator_up:
            self._set_row(self._nodes_row, snap.nodes > 0, str(snap.nodes))
        else:
            self._set_row(self._nodes_row, False, "—")

        running = snap.coordinator_up or snap.agent_running
        self._stop.configure(state=tk.NORMAL if running and not self._busy else tk.DISABLED)

    def _set_row(self, row: dict, ok: bool, text: str) -> None:
        row["dot"].delete("all")
        color = OK if ok else LINE
        row["dot"].create_oval(1, 1, 9, 9, fill=color, outline=color)
        row["value"].configure(text=text)

    def _on_close(self) -> None:
        self._persist()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run(launcher: Optional[Launcher] = None) -> None:
    LauncherWindow(launcher).run()
