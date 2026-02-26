# After matching: the 5-statement block in run_migration is replaced by a call
# to the existing _setup_logger() helper.
def _setup_logger():
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)


def init_app():
    _setup_logger()
    # other initialization omitted


def run_migration(db, plan):
    _setup_logger()
    for step in plan.steps:
        db.execute(step.sql)
    db.commit()
