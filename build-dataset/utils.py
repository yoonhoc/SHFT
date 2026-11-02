import argparse
import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


def setup_logging(log_file: str | Path, level: int = logging.INFO) -> None:
    """루트 로거에 log_file 로 회전 저장하는 파일 핸들러를 붙인다.

    각 모듈은 `logging.getLogger(__name__)` 으로 레코드만 emit 하고, propagate 를
    통해 여기서 설정한 루트 핸들러로 모인다.
    """
    root = logging.getLogger()
    root.setLevel(level)
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s][%(threadName)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def save_args(args: argparse.Namespace, path: Path) -> None:
    """실행 인자를 json 으로 남긴다 (같은 단계를 다시 돌릴 때 설정 확인용)."""
    path.write_text(json.dumps(vars(args), indent=2))


class Tee:
    """sys.stdout·sys.stderr 자리에 끼워, 쓰는 글을 원래 스트림과 파일에 같이 쓴다."""

    def __init__(self, stream: TextIO, file: TextIO):
        self.stream = stream
        self.file = file

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def tee_output(path: str | Path) -> None:
    """stdout·stderr (진행 막대, print, 오류) 를 화면과 path 에 같이 쓴다.

    nohup 으로 띄워도 실행 폴더에서 진행 상황을 볼 수 있게 한다. 실행 명령에서 따로 리다이렉트하지 않아도 된다.
    """
    file = open(path, "a", encoding="utf-8")
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, file)
    sys.stderr = Tee(sys.stderr, file)
    # import 때 이미 원래 stderr 를 잡은 로그 핸들러(transformers 등)도 파일에 쓰게 바꾼다
    loggers = [logging.getLogger()]
    for item in logging.root.manager.loggerDict.values():
        if isinstance(item, logging.Logger):
            loggers.append(item)
    for item in loggers:
        for handler in item.handlers:
            if type(handler) is logging.StreamHandler and handler.stream is original_stderr:
                handler.setStream(sys.stderr)
