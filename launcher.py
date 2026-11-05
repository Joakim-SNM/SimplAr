#!/usr/bin/env python3
"""SimplAr launcher: starts each tool and shows the Methods & Reference document."""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QMessageBox, QFileDialog, QDialog, QTextBrowser
)
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QUrl
from PyQt5.QtGui import QFont, QColor, QPalette, QDesktopServices

from about_text import build_about_html
import data_folders as dfold


# (button label, script path relative to this folder), in workflow order
TOOLS = [
    ("Data converter",     ("1_data_converter", "data_converter.py")),
    ("Flux monitors",      ("2_flux_monitors", "flux_monitors.py")),
    ("Flux regression",    ("3_flux_regression", "flux_regression.py")),
    ("Plateau & isochron", ("4_plateau_isochron", "plateau_isochron.py")),
    ("Age distribution",   ("5_age_distribution", "age_distribution.py")),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


class ToolLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self._procs = []          # every running tool, stopped in closeEvent
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(36, 28, 36, 32)
        layout.setSpacing(10)

        heading = QLabel("SimplAr")
        heading.setAlignment(Qt.AlignCenter)
        heading.setFont(QFont("Georgia", 30))
        heading.setObjectName("heading")
        layout.addWidget(heading)

        rule = QLabel()
        rule.setObjectName("rule")
        rule.setFixedHeight(1)
        layout.addWidget(rule)
        layout.addSpacing(8)

        for label, rel in TOOLS:
            btn = QPushButton(label.replace('&', '&&'))   # '&' is not a shortcut
            btn.setFont(QFont("Georgia", 14))
            btn.setMinimumHeight(46)
            if rel[0] == "1_data_converter":
                btn.clicked.connect(self.launch_data_converter)
            else:
                btn.clicked.connect(lambda _c=False, r=rel, n=label: self.launch(r, n))
            layout.addWidget(btn)

        layout.addSpacing(12)
        about_btn = QPushButton("About")
        about_btn.setFont(QFont("Georgia", 12))
        about_btn.setMinimumHeight(38)
        about_btn.setObjectName("aboutButton")
        about_btn.clicked.connect(self.show_about)
        layout.addWidget(about_btn)

        # where results and sessions are saved (one subfolder per tool)
        layout.addSpacing(10)
        self.data_lbl = QLabel()
        self.data_lbl.setObjectName("dataFolder")
        self.data_lbl.setWordWrap(True)
        self.data_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.data_lbl)
        row = QHBoxLayout()
        open_btn = QPushButton("Open data folder")
        open_btn.setObjectName("smallButton")
        open_btn.clicked.connect(self.open_data_folder)
        change_btn = QPushButton("Change\u2026")
        change_btn.setObjectName("smallButton")
        change_btn.clicked.connect(self.change_data_folder)
        row.addWidget(open_btn)
        row.addWidget(change_btn)
        layout.addLayout(row)
        self._show_data_folder()

        layout.addStretch()

        self.setStyleSheet("""
            QWidget {
                background-color: #ece3d0;
            }
            QLabel#heading {
                color: #4a3b29;
            }
            QLabel#rule {
                background-color: #cdbb98;
                margin: 2px 48px;
            }
            QPushButton {
                background-color: #ddccae;
                color: #3a2e1e;
                border: 1px solid #a98e63;
                border-radius: 3px;
                padding: 8px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #d3c09c;
            }
            QPushButton:pressed {
                background-color: #c2ac84;
            }
            QPushButton#aboutButton {
                background-color: transparent;
                color: #6a5533;
                border: 1px solid #c3ad86;
                font-style: italic;
            }
            QPushButton#aboutButton:hover {
                background-color: #e3d7bd;
            }
            QPushButton#smallButton {
                padding: 4px;
                font-size: 11px;
            }
            QLabel#dataFolder {
                color: #6a5533;
                font-size: 11px;
            }
        """)

        self.resize(380, 600)

    # ── data folder ────────────────────────────────────────────────────────────
    def _show_data_folder(self):
        self.data_lbl.setText(f"Data folder:\n{dfold.data_root()}")

    def open_data_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(dfold.create_all()))

    def change_data_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose the data folder",
                                                dfold.data_root())
        if path:
            dfold.set_data_root(path)
            dfold.create_all()
            self._show_data_folder()

    # ── launching ──────────────────────────────────────────────────────────────
    def _script(self, rel):
        path = os.path.join(ROOT, *rel)
        if not os.path.exists(path):
            QMessageBox.critical(self, "File missing",
                                 f"{os.path.basename(path)} not found at:\n{path}")
            return None
        return path

    def _start(self, proc, path, args, label):
        proc.setWorkingDirectory(os.path.dirname(path))
        proc.finished.connect(lambda *_a, p=proc: self._forget(p))
        proc.start(sys.executable, args)
        self._procs.append(proc)
        if not proc.waitForStarted(3000):
            QMessageBox.critical(self, "Launch error", f"Failed to start {label}")

    def _forget(self, proc):
        if proc in self._procs:
            self._procs.remove(proc)

    def launch(self, rel, label):
        """Start one tool as its own process, in its own folder."""
        path = self._script(rel)
        if path is None:
            return
        proc = QProcess(self)

        def _closed(code, _status):
            if code != 0:
                QMessageBox.warning(self, "Closed", f"{label} closed (exit code {code}).")
        proc.finished.connect(_closed)
        self._start(proc, path, [path], label)

    def launch_data_converter(self):
        """The converter asks for its input and output here, then shows its own
        mapping dialog; it runs as a QProcess so the launcher stays responsive."""
        path = self._script(TOOLS[0][1])
        if path is None:
            return
        input_path, _ = dfold.get_open_file_name(
            self, "Pick an ArArCALC or Mass Spec export (.xlsx)", "",
            "Excel files (*.xlsx *.xls)", tool='data_converter')
        if not input_path:
            return
        output_dir = dfold.get_existing_directory(self, "Pick an output folder",
                                                  tool='data_converter')
        if not output_dir:
            return

        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES')
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        def _done(code, _status, p=proc):
            output = bytes(p.readAll()).decode('utf-8', 'replace').strip()
            if code == 0:
                QMessageBox.information(self, "Done", output or "Conversion finished.")
            else:
                QMessageBox.warning(self, "Conversion error",
                                    f"Exited with code {code}.\n\n{output}")
        proc.finished.connect(_done)
        self._start(proc, path, ['-u', path, input_path, output_dir], "the data converter")

    # ── About / Methods document ─────────────────────────────────────────────────
    def show_about(self):
        dlg = AboutDialog(self)
        dlg.exec_()

    def closeEvent(self, event):
        for proc in list(self._procs):
            if proc and proc.state() != QProcess.NotRunning:
                proc.terminate()
                if not proc.waitForFinished(3000):
                    proc.kill()
        event.accept()


class AboutDialog(QDialog):
    """Full methods-and-reference document with a clickable table of contents."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SimplAr: Methods & Reference")
        self.resize(920, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        # equation and screenshot images are referenced by bare filename
        browser.setSearchPaths([
            dfold.cache_dir('eqcache'),
            os.path.join(ROOT, 'docs', 'img'),
            ROOT,
        ])
        browser.setStyleSheet("QTextBrowser { background:#ffffff; color:#26211a; border:none; }")
        # Force a light palette on the browser so macOS dark/"night" mode can't
        # swap the default text color to white on top of the white background.
        about_palette = browser.palette()
        about_palette.setColor(QPalette.Base, QColor('#ffffff'))
        about_palette.setColor(QPalette.Text, QColor('#26211a'))
        browser.setPalette(about_palette)
        browser.setHtml(build_about_html(ROOT))
        layout.addWidget(browser)

        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 10)
        top_btn = QPushButton("↑ Back to top")
        top_btn.clicked.connect(lambda: browser.scrollToAnchor("top"))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bar.addWidget(top_btn)
        bar.addStretch()
        bar.addWidget(close_btn)
        layout.addLayout(bar)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SimplAr")
    launcher = ToolLauncher()
    launcher.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()