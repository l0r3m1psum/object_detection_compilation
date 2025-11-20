/* Compiler and platform detection ********************************************/

// https://stackoverflow.com/a/42040445
// https://stackoverflow.com/a/11351171
#if defined(__GNUC__)
	// clang masquerades as GCC
	#if defined(_WIN32)
		#define PLATFORM_WIN32
	#elif defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
		#define PLATFORM_POSIX
	#else
		#error "unknown platform"
	#endif
#elif defined(_MSC_VER)
	#define PLATFORM_WIN32
#else
	#error "unknown compiler"
#endif

/* Platform layer *************************************************************/

#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#if defined(PLATFORM_WIN32)
	#include <winbase.h>
	#include <processthreadsapi.h>
	#include <sysinfoapi.h>
	#define EXE_SUFFIX ".exe"
#elif defined(PLATFORM_POSIX)
	#include <unistd.h>
	#include <spawn.h>
	#include <errno.h>
	#include <fcntl.h>
	extern char **environ;
	#include <sys/errno.h>
	#define EXE_SUFFIX ""
#else
	#error
#endif

// https://eklitzke.org/path-max-is-tricky
// https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
enum {
	DIR_STACK_MAX_HEIGHT = 8,
	DIR_STACK_MAX_WIDTH = 1024,
};

static struct {
	char data[DIR_STACK_MAX_HEIGHT][DIR_STACK_MAX_WIDTH];
	int pos;
} dir_stack;

// https://learn.microsoft.com/it-it/windows/win32/api/errhandlingapi/nf-errhandlingapi-setlasterror
enum platform_error {
	ERROR_OK,
	ERROR_DIR_STACK_OVERFLOW,
	ERROR_DIR_STACK_UNDERFLOW,
	ERROR_STR_TOO_BIG,
	ERROR_COMMAND_FAILED,
} error;

static void
platform_seterr(enum platform_error err) {
	error = err;
}

static void
platform_reseterr(void) {
	error = ERROR_OK;
}

static bool
platform_test_error() {
#if defined(PLATFORM_WIN32)
	return GetLastError() == ERROR_SUCCESS && error == ERROR_OK;
#elif defined(PLATFORM_POSIX)
	return (errno == 0) && (error == ERROR_OK);
#else
	#error
#endif
}

static bool
platform_run(const char **cmd) {
	// TODO: log the command
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		// TODO: covert args to a single string.
		res = CreateProcessA(
			[in, optional]      LPCSTR                lpApplicationName,
			[in, out, optional] LPSTR                 lpCommandLine,
			[in, optional]      LPSECURITY_ATTRIBUTES lpProcessAttributes,
			[in, optional]      LPSECURITY_ATTRIBUTES lpThreadAttributes,
			[in]                BOOL                  bInheritHandles,
			[in]                DWORD                 dwCreationFlags,
			[in, optional]      LPVOID                lpEnvironment,
			[in, optional]      LPCSTR                lpCurrentDirectory,
			[in]                LPSTARTUPINFOA        lpStartupInfo,
			[out]               LPPROCESS_INFORMATION lpProcessInformation
		);
#elif defined(PLATFORM_POSIX)
		pid_t pid = 0;
		errno = posix_spawnp(
			&pid,
			cmd[0],
			NULL, // ignore file actions
			NULL, // ignore attributes
			cmd,
			environ
		);
		if (errno == 0) {
			int status = 0;
			res = waitpid(pid, &status, 0) == 0;
			if (WIFEXITED(status)) {
				if (WEXITSTATUS(status) != 0) {
					printf("exited, status=%d\n", WEXITSTATUS(status));
					fprintf(stderr, "\"%s\" exited with non zero return code\n", cmd[0]);
					platform_seterr(ERROR_COMMAND_FAILED);
				}
			} else if (WIFSIGNALED(status)) {
				printf("killed by signal %d\n", WTERMSIG(status));
			} else if (WIFSTOPPED(status)) {
				printf("stopped by signal %d\n", WSTOPSIG(status));
			} else if (WIFCONTINUED(status)) {
				printf("continued\n");
			}
		} else {
			fprintf(stderr, "\"%s\" could not be spawned\n", cmd[0]);
			platform_seterr(ERROR_COMMAND_FAILED);
			res = false;
		}
#else
	#error
#endif
	}
	return res;
}

static bool
platform_cd(const char *dir) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		res = SetCurrentDirectory(dir) == 0;
#elif defined(PLATFORM_POSIX)
		res = chdir(dir) == 0;
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not cd to \"%s\"", dir);
		}
	}
	return res;
}

static bool
platform_mkdir(const char *dir) {
	bool res = false;
	// TODO: wrap in setlocal enableextensions
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
	// TODO: CreateDirectory(subpath, NULL) (call iteratively until all sub-directory are created)
		res = platform_run((const char *[]){"md", dir, NULL});
#elif defined(PLATFORM_POSIX)
	// TODO: mkdir(subpath, 0777) (call iteratively until all sub-directory are created)
		res = platform_run((const char *[]){"mkdir", "-p", dir, NULL});
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not mkdir \"%s\"\n", dir);
		}
	}
	return res;
}

static bool
platform_getcwd(char *buf, size_t len) {
	bool res = false;
	// While this function is innocuous we avoid polluting the global error.
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		res = GetCurrentDirectory(buf, len) != 0;
#elif defined(PLATFORM_POSIX)
		res = getcwd(buf, len) != NULL;
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not getcwd\n");
		}
	}
	return res;
}

static bool
platform_copy(const char *from, const char *to) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		res = CopyFile(from, to, 0) == 0;
#elif defined(PLATFORM_POSIX)
		// sendfile
		// splice (linux)
		// reflink (linux) https://man7.org/linux/man-pages/man2/ioctl_ficlonerange.2.html
		// clonefile (macos)
		res = platform_run((const char *[]){"cp", "-f", from, to, NULL});
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not copy \"%s\" to \"%s\"\n", from, to);
		}
	}
	return res;
}

static bool
platform_setenv(const char *var, const char *value) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		res = SetEnvironmentVariable(var, value);
#elif defined(PLATFORM_POSIX)
		res = setenv(var, value, 1) == 0;
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not set envvar \"%s\" to \"%s\"\n", var, value);
		}
	}
	return res;
}

static bool
platform_unsetenv(const char *var) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		res = SetEnvironmentVariable(var, NULL);
#elif defined(PLATFORM_POSIX)
		res = unsetenv(var) == 0;
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not unset envvar \"%s\"\n", var);
		}
	}
}

static bool
platform_getenv(const char *var, char *out, size_t len) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		res = GetEnvironmentVariable(var, out, len) != 0;
#elif defined(PLATFORM_POSIX)
		char *tmp = getenv(var);
		if (tmp) {
			res = strlcpy(out, tmp, len) < len;
		} else {
			res = false;
		}
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not get envvar \"%s\"\n", var);
		}
	}
	return res;
}

static bool
platform_mklink(const char *from, const char *to) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		res = CreateHardLink(to, from, NULL) != 0;
#elif defined(PLATFORM_POSIX)
		res = link(from, to) == 0;
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not hard link \"%s\" to \"%s\"\n", from, to);
		}
	}
	return res;
}

// https://stackoverflow.com/a/150971
static bool
platform_getnproc(int *nproc) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		SYSTEM_INFO sysinfo;
		GetSystemInfo(&sysinfo);
		*nproc = sysinfo.dwNumberOfProcessors;
		res = true;
#elif defined(PLATFORM_POSIX)
		long tmp = sysconf(_SC_NPROCESSORS_ONLN);
		if (tmp != -1) {
			*nproc = tmp;
			res = true;
		} else {
			res = false;
		}
#else
	#error
#endif
		if (!res) {
			fprintf(stderr, "could not get number of processors\n");
		}
	}
	return res;
}

static bool
platform_pushd(const char *dir) {
	if (platform_test_error()) {
		if (dir_stack.pos < DIR_STACK_MAX_HEIGHT-1) {
			char *buf = dir_stack.data[dir_stack.pos++];
			platform_getcwd(buf, sizeof *dir_stack.data);
		} else {
			fprintf(stderr, "Cannot push \"%s\", directory stack is full\n", dir);
			platform_seterr(ERROR_DIR_STACK_OVERFLOW);
		}
	}
	platform_mkdir(dir);
	return platform_cd(dir);
}

static bool
platform_popd(void) {
	bool res = false;
	if (platform_test_error()) {
		const char *path = "";
		if (dir_stack.pos > 0) {
			path = dir_stack.data[--dir_stack.pos];
		} else {
			fprintf(stderr, "Cannot pop directory stack is empty\n");
			platform_seterr(ERROR_DIR_STACK_OVERFLOW);
		}
		res = platform_cd(path);
	}
	return res;
}

static bool
platform_writefile(const char *str, const char *path) {
	bool res = false;
	if (platform_test_error()) {
#if defined(PLATFORM_WIN32)
		#error
#elif defined(PLATFORM_POSIX)
		int fd = open(path, O_WRONLY | O_APPEND | O_CREAT);
		if (fd != -1) {
			if (write(fd, str, strlen(str)) == -1) {
				perror("write");
			}
		} else {
			perror("open");
		}
#else
	#error
#endif
	}
	return res;
}

/******************************************************************************/

#define CONFIG "RelWithDebInfo"
#define TARGET "install"
#define INSTALLDIR "/Users/diegobellani/Developer/object_detection_compilation"
#define NPROC "12"

static bool
cmake_build(void) {
	return platform_run((const char *[]){
		"cmake", "--build", ".", "--config", CONFIG, "--target", TARGET, "--parallel", NPROC,
	NULL});
}

static const char *
leak_str(const char *fmt, ...) {
	char *buf = malloc(1024);
	if (buf) {
		va_list ap;
    	va_start(ap, fmt);
		bool ok = vsnprintf(buf, 1024, fmt, ap) < 1024;
		va_end(ap);
		if (!ok) {
			platform_seterr(ERROR_STR_TOO_BIG);
		}
	}
	return buf;
}

int main(int argc, char* argv[]) {
	struct { bool no_rebuild, help, run; } args = {0};

	for (int i = 1; i < argc; i++) {
		const char* arg = argv[i];
			 if (!strcmp(arg, "no_rebuild")) args.no_rebuild = true;
		else if (!strcmp(arg, "help")) args.help = true;
		else if (!strcmp(arg, "run")) args.run = true;
	}

	if (!args.no_rebuild) {
		printf("Rebuilding...\n");
		size_t cmd_size = 1024;
		char* cmd = calloc(1, cmd_size);
		strcat(cmd, "clang -g build.c -o build && ./build no_rebuild");
		for (int i = 1; i < argc; i++) {
			const char* arg = argv[i];
			if (strcmp(arg, "no_rebuild")) {
				strlcat(cmd, " ", cmd_size);
				strlcat(cmd, arg, cmd_size);
			}
		}
		return system(cmd);
		// platform_run((const char *[]){"clang", "-g", "build.c", "-o", "build", NULL})
		// platform_run((const char *[]){"build", "no_rebuild", NULL})
	}

	char installdir[1024];
	platform_getenv("installdir", installdir, sizeof installdir);
	int nproc = 0;
	platform_getnproc(&nproc);

	// CONFIG TARGET

	{
		platform_pushd("submodules/tvm/3rdparty/dlpack/build");
			platform_run((const char *[]){
				"cmake",
				"-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DBUILD_MOCK=no",
				"-DCMAKE_INSTALL_PREFIX=" INSTALLDIR "/Programs/dlpack/",
				"-DCMAKE_BUILD_TYPE=" CONFIG, "..",
			NULL});
			cmake_build();
		platform_popd();

		platform_pushd("submodules/tvm/3rdparty/dmlc-core/build");
			platform_run((const char *[]){
				"cmake",
				"-DUSE_OPENMP=OFF",
				"-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DBUILD_MOCK=no",
				"-DCMAKE_INSTALL_PREFIX=" INSTALLDIR "/Programs/dmlc/",
				"-DCMAKE_BUILD_TYPE=" CONFIG, "..",
			NULL});
			cmake_build();
		platform_popd();

		platform_pushd("submodules/safetensors-cpp/build");
			platform_run((const char *[]){
				"cmake",
				"-DCMAKE_INSTALL_PREFIX=" INSTALLDIR "/Programs/safetensors",
				"-DCMAKE_BUILD_TYPE=" CONFIG, "..",
			NULL});
			cmake_build();
		platform_popd();

		platform_pushd("submodules/cpython");
#if defined(PLATFORM_WIN32)
			platform_pushd("PCbuild");
				/* TODO: This should work with the -E flag i.e. we should include
				 * all python's essential dependencies as submodules. To do so add
				 * flags:
				 *     --no-ctypes --no-ssl --no-tkinter
				 * and download
				 *     bzip2 sqlite xz zlib
				 * as get_externals.bat does.
				 */
				if (strcmp(TARGET, "clean") == 0) {
					platform_run((const char *[]){"clean.bat", NULL});
				} else {
					platform_run((const char *[]){"build.bat", NULL});
				}
			platform_popd();
			if (strcmp(TARGET, "install") == 0) {
				platform_run((const char *[]){
					"python.bat", "PC/layout",
					"--include-stable", "--include-pip", "--include-pip-user",
					"--include-distutils", "--include-venv", "--include-dev",
					"--copy", INSTALLDIR "/Programs/Python/",
				NULL});
			}
#elif defined(PLATFORM_POSIX)
			platform_unsetenv("TMPDIR"); // On macOS is already defined.
			// Autoconf cannot handle spaces in pahts... https://stackoverflow.com/a/16202169
			platform_run((const char *[]){"./configure", "CC=clang", "-C", "--prefix", INSTALLDIR "/Programs/Python/", NULL});
			platform_run((const char *[]){"dot_clean", ".", NULL});
			platform_run((const char *[]){"make", TARGET, "-j", NPROC, NULL});
#else
	#error
#endif
		platform_popd();

		platform_pushd("submodules/llvm-project/llvm/build");
			platform_run((const char *[]){
				"cmake",
				"-DCMAKE_INSTALL_PREFIX=" INSTALLDIR "/Programs/LLVM",
				"-DLLVM_ENABLE_PROJECTS=clang",
				"-DLLVM_INCLUDE_TESTS=OFF",
				"-DCMAKE_BUILD_TYPE=" CONFIG, "..",
			NULL});
			cmake_build();
			platform_mklink(
				INSTALLDIR "/Programs/LLVM/bin/clang" EXE_SUFFIX,
				INSTALLDIR "/Programs/LLVM/bin/gcc" EXE_SUFFIX
			);
		platform_popd();
	}

	return 0;

	{
		platform_pushd("submodules/tvm");
			platform_pushd("build");
				platform_copy("../cmake/config.cmake", ".");
				platform_copy("../../../vtar/VTAR.cmake", ".");
				platform_writefile(
					// msbuild ignores CMAKE_BUILD_TYPE
					// "set(CMAKE_BUILD_TYPE " CONFIG ")\n"
					"set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")\n"
					"set(HIDE_PRIVATE_SYMBOLS ON)\n"
					"set(USE_VTA_FSIM ON)\n"
					"set(Python_ROOT_DIR \"" INSTALLDIR "\")\n"
					"set(Python_FIND_REGISTRY NEVER)\n"
					"set(Python_FIND_STRATEGY LOCATION)\n"
					/* When LLVM is compiled in debug mode this is needed when compiling TVM in Release or RelWithDebInfo mode
					 * echo set(USE_MSVC_MT ON) >> config.cmake
					 * echo add_compile_options("/MT") >> config.cmake || goto :exit
					 */
					"if (MSVC) add_compile_options(\"/MDd\") endif()\n"
					,
					"config.cmake"
				);
				platform_run((const char *[]){
					"cmake", "-DCMAKE_INSTALL_PREFIX=" INSTALLDIR "/Programs/TVM", "..",
				NULL});
				static char cwdbuf[1024];
				platform_getcwd(cwdbuf, sizeof cwdbuf);
				// Needed because otherwise LINK.EXE cannot find tvm.lib
				platform_setenv("LINK", leak_str("/LIBPATH:%s/%s /LIBPATH:%s/Programs/Python/libs", cwdbuf, CONFIG, INSTALLDIR));
				platform_setenv("INCLUDE", leak_str("%s/../ffi/include", cwdbuf));
				platform_setenv("UseEnv", "true");
				cmake_build();
				platform_setenv("TVM_LIBRARY_PATH", leak_str("%s/%s", cwdbuf, CONFIG));
			platform_popd();
			if (strcmp(TARGET, "clean") != 0) {
				platform_setenv("LINK", "/LIBPATH:" INSTALLDIR "/Programs/Python/libs /LIBPATH:" INSTALLDIR "/Programs/TVM/lib");
				// --no-build-isolation makes it faster
				platform_run((const char *[]){
					"python",
					"-m", "pip", "install", "--no-build-isolation", "--no-index",
					"--find-links", INSTALLDIR "/Programs/wheelhouse", "./python",
				NULL});
				platform_run((const char *[]){
					"python",
					"-c", "import tvm; print(tvm.__file__); print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))",
				NULL});
			}
		platform_popd();
	}

	{
		// TODO: port build_runtime.sh
	}

	return EXIT_SUCCESS;
}
