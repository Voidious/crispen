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
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.DEBUG)
    for step in plan.steps:
        db.execute(step.sql)
    db.commit()
