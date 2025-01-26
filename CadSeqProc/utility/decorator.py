from functools import wraps
import tracemalloc
import time
import gc
import torch
import gc
import datetime
from contextlib import ContextDecorator
from rich import print
# <---------------- Custom Decorators ---------------->

"""
def my_decorator_func(func):

    def wrapper_func():
        # Do something before the function.
        func()
        # Do something after the function.
        # May return the result of the func()
    return wrapper_func
"""


def convert_seconds_to_minutes_and_hours(seconds):
    """
    Converts seconds to minutes and hours.

    Args:
        seconds (int): The number of seconds.

    Returns:
        tuple: The minutes and hours.

    """

    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60

    if minutes==0:
        if seconds<0.1:
            return f"{seconds*1000} ms"
        return f"{seconds} seconds"
    elif hours==0:
        return f"{minutes} minutes"
    else:
        return f"{hours} hours {minutes} minutes"

def timeit(func):
    # Decorator for calculating time
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {convert_seconds_to_minutes_and_hours(total_time)}')
        return result
    return timeit_wrapper


def log_datetime(func):
    """Log the date and time of a function"""
    @wraps(func)
    def log_datetime_wrapper(*args,**kwargs):
        startInfo=f'Function: {func.__name__} \nRun on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        result=func(*args,**kwargs)
        print(startInfo)
        return result
        print(f'{"-"*30}')
    return log_datetime_wrapper


def measure_performance(func):
    '''Measure performance of a function'''

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        result=func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        finish_time = time.perf_counter()
        print(f'{"-"*40}')
        print(f'Function: {func.__name__}')
        print(f'Method: {func.__doc__}')
        print(f'Memory usage:\t\t {current / 10**6:.6f} MB \n'
              f'Peak memory usage:\t {peak / 10**6:.6f} MB ')
        print(f'Time elapsed : {convert_seconds_to_minutes_and_hours(finish_time - start_time)}')
        print(f'{"-"*40}')
        tracemalloc.stop()
        return result
    return wrapper



def gpu_memory_usage(func):
    """
    Decorator that prints the GPU memory usage before and after a function is called.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    """

    def wrapper(*args, **kwargs):
        start_memory = torch.cuda.memory_allocated()
        func(*args, **kwargs)
        end_memory = torch.cuda.memory_allocated()
        print("GPU memory usage: " + str(end_memory - start_memory))

    return wrapper



# Code from https://gist.github.com/MarkTension/4783697ebd5212ba500cdd829b364338
# pytorch method to find number of tensors in the graph
def get_n_tensors():
    tensors= []
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
                tensors.append(obj)
        except:
            pass
        return len(tensors)
  
# this is our context decorator
class check_memory_leak_context(ContextDecorator):
    def __enter__(self):
        self.start = get_n_tensors()
        return self
    def __exit__(self, *exc):
        self.end = get_n_tensors()
        increase = self.end - self.start
        
        if increase > 0:
                print(f"num tensors increased with"\
                    f"{self.end - self.start} !")
        else:
                print("no added tensors")
        return False

"""
Decorator utilities for enhanced geometry system.
"""

import functools
import time
from typing import Any, Callable, TypeVar, cast
from .logger import setup_logger

logger = setup_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])

def log_execution(func: F) -> F:
    """Log function execution time and parameters."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return cast(F, wrapper)

def validate_parameters(func: F) -> F:
    """Validate function parameters."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get function parameter names
        params = func.__code__.co_varnames[:func.__code__.co_argcount]
        
        # Check for None values
        for name, value in zip(params, args):
            if value is None:
                logger.warning(f"Parameter {name} is None in {func.__name__}")
        
        for name, value in kwargs.items():
            if value is None:
                logger.warning(f"Parameter {name} is None in {func.__name__}")
        
        return func(*args, **kwargs)
    return cast(F, wrapper)

def cache_result(func: F) -> F:
    """Cache function results."""
    cache: dict = {}
    
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            logger.debug(f"Cached result for {func.__name__}")
        else:
            logger.debug(f"Using cached result for {func.__name__}")
        
        return cache[key]
    return cast(F, wrapper)

def retry_on_error(max_attempts: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """Retry function on error."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Max retry attempts ({max_attempts}) reached for {func.__name__}")
                        raise
                    logger.warning(f"Attempt {attempts} failed for {func.__name__}: {str(e)}")
                    time.sleep(delay)
            return None  # Should never reach here
        return cast(F, wrapper)
    return decorator

def deprecated(func: F) -> F:
    """Mark function as deprecated."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.warning(f"Function {func.__name__} is deprecated")
        return func(*args, **kwargs)
    return cast(F, wrapper)
