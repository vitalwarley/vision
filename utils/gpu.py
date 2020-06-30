import multiprocessing
from functools import wraps
import tensorflow as tf

def set_gpu():
    """Set dynamic allocation."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# FIXME: why doesn't work?
        # -> pickle.PicklingError: Can't pickle <function at ...>: ...
def run_and_release(func):

    @wraps(func)
    def parallel_wrapper(output, *argv, **kwargs):
        results = func(*argv, **kwargs)
        output['results'] = results

    def outer_wrapper(*argv, **kwargs):
        with multiprocessing.Manager() as manager:
            output = manager.dict()
            args = (output, ) + argv
            proc = multiprocessing.Process(target=parallel_wrapper, args=args,
                                           kwargs=kwargs)
            proc.start()
            proc.join()
            results = output.get('results', [])

        return results

    return outer_wrapper
