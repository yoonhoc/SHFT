import logging
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Runner(ABC):
    @property
    @abstractmethod
    def max_workers(self) -> int: ...

    @abstractmethod
    def create_jobs(self) -> list[BaseModel]: ...

    @abstractmethod
    def process_job(self, job: BaseModel) -> None: ...

    def run(self, jobs: list[BaseModel]):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_job, job) for job in jobs]
            done = as_completed(futures)
            pbar = tqdm(done, total=len(futures), dynamic_ncols=True)
            for future in pbar:
                try:
                    future.result()
                except Exception:
                    print(traceback.format_exc())
