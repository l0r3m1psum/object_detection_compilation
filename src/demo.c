// cl /nologo demo.c User32.lib Vfw32.lib
#include <windows.h>
#include <vfw.h>
#include <stdio.h>

// https://learn.microsoft.com/en-us/windows/win32/multimedia/using-video-capture

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_CLOSE:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

TCHAR gachBuffer[100];
DWORD gdwFrameNum = 0;

LRESULT PASCAL FrameCallbackProc(HWND hWnd, LPVIDEOHDR lpVHdr)
{
    if (!hWnd)
        return FALSE;

    // _stprintf_s(gachBuffer, TEXT("Preview frame# %ld "), gdwFrameNum++);
    // SetWindowText(hWnd, gachBuffer);
    for (int i = 0; i < 10000; i++) lpVHdr->lpData[i] = 0;
    return (LRESULT)TRUE;
}

int APIENTRY WinMain(HINSTANCE hInst, HINSTANCE hInstPrev, PSTR cmdline, int cmdshow)
{
    WNDCLASS wc = {
        .style = 0,
        .lpfnWndProc = WindowProc,
        .cbClsExtra = 0, .cbWndExtra = 0,
        .hInstance = GetModuleHandle(NULL),
        .hIcon = NULL,
        .hCursor = NULL,
        .hbrBackground = NULL,
        .lpszMenuName = NULL,
        .lpszClassName = TEXT("CaptureWindowClass"),
    };
    ATOM cls = RegisterClass(&wc);
    if (cls == 0) {
        abort();
    }

    HANDLE hWndMain = CreateWindow(
        (LPCTSTR)cls,
        TEXT("WebCam Viewer"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        640, 480,
        /* hWndParent */ NULL,
        /* hMenu */ NULL,
        hInst,
        /* lpParam */ NULL
    );
    ShowWindow(hWndMain, SW_SHOW);

    int nID = 32;
    HANDLE hWndC = capCreateCaptureWindow(
        TEXT("My Capture Window"),   // window name if pop-up 
        WS_CHILD | WS_VISIBLE,       // window style 
        0, 0, 640, 480,              // window position and dimensions
        (HWND)hWndMain,
        (int)nID /* child ID */ // ????
    );

    capSetCallbackOnFrame(hWndC, FrameCallbackProc);

    capDriverConnect(hWndC, 0); // FIXME: how do I check for errors?

    capPreviewRate(hWndC, 66);
    capPreview(hWndC, TRUE);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}
