import ctypes.util
import signal
import os

# https://www.cocoawithlove.com/2008/03/break-into-debugger.html#:~:text=Debugger()%20and%20DebugStr()
# https://developer.apple.com/documentation/xcode/sigtrap_sigill

def os_breakpoint():
	if os.name == 'nt':
		ctypes.cdll.kernel32.DebugBreak()
	elif os.name == 'posix':
		libc_name = ctypes.util.find_library('c')
		if libc_name is not None:
			getattr(ctypes.CDLL(libc_name), 'raise')(signal.SIGTRAP)
		else:
			raise FileNotFoundError("cannot find libc")
	else:
		raise Exception("os.name = '%s' not supported" % (os.name,))
