from crispen.config import CrispenConfig
from crispen.engine import run_engine


def _run(changed):
    return list(run_engine(changed, config=CrispenConfig(min_tuple_size=3)))
