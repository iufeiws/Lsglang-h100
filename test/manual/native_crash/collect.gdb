set pagination off
set confirm off
set print thread-events off
set print elements 32
set startup-with-shell off
set disable-randomization off
set detach-on-fork off
set follow-fork-mode parent
set follow-exec-mode same
set schedule-multiple on
handle SIGPIPE nostop noprint pass
handle SIGCHLD nostop noprint pass
catch signal SIGSEGV
commands
silent
python
import gdb
thread = gdb.selected_thread()
gdb.write("\n=== NATIVE-FAULT pid=%s ptid=%s ===\n" % (gdb.selected_inferior().pid, thread.ptid if thread else None))
# One failed read (e.g. a corrupted PC) must not suppress the remaining evidence.
for command in ("p $_siginfo", "p/x $pc", "info symbol $pc", "bt 40",
                "info registers", "info sharedlibrary", "info proc mappings",
                "info inferiors", "info threads", "thread apply all bt 12"):
    gdb.write("\n--- " + command + " ---\n")
    try:
        gdb.execute(command)
    except gdb.error as exc:
        gdb.write("[unavailable] " + str(exc) + "\n")
gdb.write("\n=== END-NATIVE-FAULT ===\n")
end
quit 139
end
python
# A shell launcher can exit while its scheduler children remain alive.
# Keep all fork/exec descendants scheduled until they exit or hit the catchpoint.
import gdb
gdb.execute("run")
while any(inferior.pid for inferior in gdb.inferiors()):
    live = next(inferior for inferior in gdb.inferiors() if inferior.pid)
    gdb.execute("inferior %d" % live.num)
    gdb.execute("continue")
end
