// T-GPLR-16 diagnostic: dump symbolized stacks of all threads of a hung process.
// Usage: gplr16-dump.exe <pid>
// Built with MSVC (cl) from VS Build Tools; links dbghelp.
#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>
#include <stdio.h>
#include <stdlib.h>

#pragma comment(lib, "dbghelp.lib")

typedef BOOL (WINAPI *PGetThreadContextFn)(HANDLE, LPCONTEXT);

static void dump_thread(HANDLE hProc, DWORD tid, HANDLE hThread) {
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    if (!GetThreadContext(hThread, &ctx)) {
        printf("tid %lu: GetThreadContext failed %lu\n", tid, GetLastError());
        return;
    }
    STACKFRAME64 frame{};
    DWORD imageType;
#ifdef _WIN64
    imageType = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset = ctx.Rip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp;
    frame.AddrStack.Mode = AddrModeFlat;
#else
    imageType = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset = ctx.Eip;
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Ebp;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = ctx.Esp;
    frame.AddrStack.Mode = AddrModeFlat;
#endif
    printf("=== tid %lu ===\n", tid);
    for (int i = 0; i < 64; ++i) {
        if (!StackWalk64(imageType, hProc, hThread, &frame, &ctx, nullptr,
                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
            break;
        }
        if (frame.AddrPC.Offset == 0) {
            break;
        }
        DWORD64 modBase = SymGetModuleBase64(hProc, frame.AddrPC.Offset);
        char modName[MAX_PATH] = "<unknown>";
        if (modBase) {
            IMAGEHLP_MODULE64 mi{};
            mi.SizeOfStruct = sizeof(mi);
            if (SymGetModuleInfo64(hProc, modBase, &mi)) {
                strncpy_s(modName, mi.ImageName, _TRUNCATE);
                const char * slash = strrchr(modName, '\\');
                if (slash) { memmove(modName, slash + 1, strlen(slash)); }
            }
        }
        char symBuf[sizeof(SYMBOL_INFO) + 256]{};
        PSYMBOL_INFO sym = (PSYMBOL_INFO) symBuf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 disp = 0;
        const char * name = "?";
        char nameBuf[256] = "";
        if (SymFromAddr(hProc, frame.AddrPC.Offset, &disp, sym)) {
            strncpy_s(nameBuf, sym->Name, _TRUNCATE);
            name = nameBuf;
        }
        printf("  #%02d %s+0x%llx [%s]\n", i, name, (unsigned long long) disp, modName);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pid>\n", argv[0]);
        return 2;
    }
    DWORD pid = (DWORD) strtoul(argv[1], nullptr, 10);
    HANDLE hProc = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!hProc) {
        fprintf(stderr, "OpenProcess(%lu) failed %lu\n", pid, GetLastError());
        return 1;
    }
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    if (!SymInitialize(hProc, nullptr, TRUE)) {
        fprintf(stderr, "SymInitialize failed %lu (continuing)\n", GetLastError());
    }
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "snapshot failed\n");
        return 1;
    }
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID != pid) { continue; }
            HANDLE hThread = OpenThread(THREAD_ALL_ACCESS, FALSE, te.th32ThreadID);
            if (!hThread) { continue; }
            if (SuspendThread(hThread) == (DWORD) -1) { CloseHandle(hThread); continue; }
            dump_thread(hProc, te.th32ThreadID, hThread);
            ResumeThread(hThread);
            CloseHandle(hThread);
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    SymCleanup(hProc);
    CloseHandle(hProc);
    return 0;
}
