import sys
import os
import time
import logging

logger = logging.getLogger(__name__)

if sys.platform != 'win32':
    logger.error("This module requires Windows")
    raise SystemExit(1)

try:
    import win32job
    import win32api
except ImportError as ex:
    logger.error("pywin32 library not found, please install it using: pip install pywin32")
    raise

def set_memory_limit(memory_limit_bytes: int) -> object:

    security_attributes = None
    job_name = f"PythonMemLimitJob_{os.getpid()}"
    job_handle = win32job.CreateJobObject(security_attributes, job_name)
    if not job_handle:
        err_code = win32api.GetLastError()
        raise RuntimeError(f"Error creating Job Object: Code {err_code} - {win32api.FormatMessage(err_code)}")

    basic_limits = {
        "PerProcessUserTimeLimit": 0,
        "PerJobUserTimeLimit": 0,
        "MinimumWorkingSetSize": 0,
        "MaximumWorkingSetSize": 0,
        "ActiveProcessLimit": 0,
        "Affinity": 0,
        "PriorityClass": 0,
        "SchedulingClass": 0,
        "LimitFlags":
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
    }

    io_counters = {
        "ReadOperationCount": 0,
        "WriteOperationCount": 0,
        "OtherOperationCount": 0,
        "ReadTransferCount": 0,
        "WriteTransferCount": 0,
        "OtherTransferCount": 0
    }

    extended_limits = {
        "BasicLimitInformation": basic_limits,
        "IoInfo": io_counters,
        "ProcessMemoryLimit": memory_limit_bytes,
        "JobMemoryLimit": 0,
        "PeakProcessMemoryUsed": 0,
        "PeakJobMemoryUsed": 0
    }

    # May fail...
    win32job.SetInformationJobObject(
        job_handle,
        win32job.JobObjectExtendedLimitInformation,
        extended_limits
    )

    process_handle = win32api.GetCurrentProcess()

    # May fail if already the process is assigned to job object
    win32job.AssignProcessToJobObject(job_handle, process_handle)

    return job_handle

def remove_memory_limit(job_handle: object) -> None:
    win32api.CloseHandle(job_handle)
