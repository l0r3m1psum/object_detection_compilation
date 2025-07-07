// cl /ZI /DUNICODE dd.c Shell32.lib /link

#include <windows.h>
#include <shellapi.h>

#define BUFFER_SIZE 512

int
main(void) {

	// FIXME: broken for some reason
#if 0
	FILE_STORAGE_INFO fileStorageInfoBuf = {0};
	GetFileInformationByHandleEx(
		CreateFile("\\\\.\\PhysicalDrive0", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL),
		FileStorageInfo,
		&fileStorageInfoBuf,
		sizeof fileStorageInfoBuf
	);
	__debugbreak();
#endif

	int numArgs = 0;
	LPTSTR *argv = CommandLineToArgvW(GetCommandLine(), &numArgs);

	HANDLE hIn = INVALID_HANDLE_VALUE;
	HANDLE hOut = INVALID_HANDLE_VALUE;
	if (argv) {
		if (numArgs == 3) {
			LPCTSTR in_path = argv[1];
			LPCTSTR out_path = argv[2];

			hIn = CreateFile(in_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
			if (hIn != INVALID_HANDLE_VALUE) {
				hOut = CreateFile(out_path, GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_FLAG_WRITE_THROUGH, NULL);
			}
		} else {
			SetLastError(ERROR_BAD_ARGUMENTS);
		}
	}

	if (hIn != INVALID_HANDLE_VALUE && hOut != INVALID_HANDLE_VALUE) {
		static BYTE buffer[BUFFER_SIZE] = {0};
		DWORD bytesRead = 0, bytesWritten = 0;
		while (ReadFile(hIn, buffer, sizeof buffer, &bytesRead, NULL) && bytesRead > 0) {
			BOOL ok = WriteFile(hOut, buffer, bytesRead, &bytesWritten, NULL);
			BOOL partialWrite = bytesWritten != bytesRead;
			if (!ok || partialWrite) {
				break;
			}
		}
	}

	DWORD err = GetLastError();
	LPTSTR lpMsgBuf = NULL;

	if (err) {
		DWORD nTCHARWritten = FormatMessage(
			/* dwFlags */ FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
			/* lpSource */ NULL,
			/* dwMessageId */ err,
			/* dwLanguageId */ 0,
			/* lpBuffer */ (LPTSTR) &lpMsgBuf, // Just to avoid a warning.
			/* nSize */ 0,
			/* Arguments */ NULL
		);
		HANDLE stdErr = GetStdHandle(STD_ERROR_HANDLE);
		WriteConsole(stdErr, lpMsgBuf, nTCHARWritten, NULL, NULL);
	}

	// LocalFree(argv);
	// LocalFree(lpMsgBuf);
	return (err);
}
