import time

class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()  # High-resolution timer
        return self  # Return self for accessing elapsed time later if needed

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time