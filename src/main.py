def _time_execution(img_path: str = "path/to/file"):

    import time
    from models.moondream import Moondream_FastVLM

    t0 = time.perf_counter()
    vlm = Moondream_FastVLM()
    t1 = time.perf_counter()
    print(f"Init took {t1 - t0:.2f} seconds")

    t2 = time.perf_counter()
    vlm.detect(img_path, "human")
    t3 = time.perf_counter()
    print(f"Detect took {t3 - t2:.2f} seconds")

    t4 = time.perf_counter()
    vlm.point(img_path, "human")
    t5 = time.perf_counter()
    print(f"Point took {t5 - t4:.2f} seconds")

    print(f"Total runtime: {t5 - t0:.2f} seconds")


def main():
    from server.core.config import Settings
    import uvicorn

    settings = Settings()
    uvicorn.run("server:app", host=settings.host, port=settings.port, reload=True)


if __name__ == "__main__":
    main()
