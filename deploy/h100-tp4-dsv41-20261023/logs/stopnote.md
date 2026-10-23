### 12.0 停止事件与数据边界（按指令停止，非崩溃）

本轮压测 07:26:58 启动，计划 `single → K1 → K4 → K8 → K16`。08:05:48 服务收到**外部 SIGTERM**，K4 档自此刻起出现 `conn_err`，K8/K16 档完全无法建立连接。

| 时刻 (+08:00) | 事件 | 原始行 |
| --- | --- | --- |
| 08:04:50 | 最后一条成功请求 | `INFO: 127.0.0.1:37296 - "POST /v1/chat/completions HTTP/1.1" 200 OK` |
| 08:05:0x | 宿主回退（4 行） | `[hf] layer 4 took over after timeout` / `layer 6` / `layer 15` / `layer 18` |
| 08:05:18 | 该档一次 decode 行 | `Decode batch, #running-req: 4, … gen throughput (token/s): 4.74` |
| 08:05:48 | **外部 SIGTERM** | `SIGTERM received. signum=None frame=None. Draining requests and shutting down...` |
| 08:05:49 | 子进程被 -15 终止 | `Subprocess detokenizer (pid=56900) crashed with exit code -15. Triggering SIGQUIT for cleanup...` |
| 08:05:49 | — | `SIGQUIT received. signum=None, frame=None. It usually means one child failed.` |
| 08:05:54 | 进程树清理 | `No live scheduler processes found; skipping py-spy and CUDA coredump.` / `kill_process_tree called: parent_pid=55964, include_parent=True, pid=55964` |

服务日志崩溃类计数（全文）：`SEGV` = 2（均为脚本头静态 `[ENV-CHANGE]` 文本）、`OOM` = 0、`Traceback` = 0、`CUDA error` = 0、`illegal` = 0、`core dumped` = 0。

由 SIGTERM 造成的档位残缺：K4 档 27 轮中 15 成功 / 12 失败（`conn_err`、`other_err`）；K8 档 24 轮全失败；K16 档 48 轮全失败（均为 `conn_err`，`Max retries exceeded … NewConnectionError`）。客户端在 K16 档结束（08:06:35）后按计划自行退出，08:06:56 落盘完毕，无残留压测进程。

测量侧纪律：本轮**未**重启服务、**未**改服务端配置/脚本/site-packages、**未**触碰 GPU0–3。08:10:33 启动的 `pid=16080` 为**另一配置**的新服务，不属本轮测量对象，本轮亦未对其发送任何请求。

**thinking 阶段未执行**：要求改长会话为 thinking(max) 并保留 reasoning 的指令到达时刻晚于压测客户端自行结束时刻（08:06:56），本轮无 `reasoning_content` 相关数据。

停止后现场快照（2026-10-23 08:12:00，GPU4/5/6/7，MiB）: 42426 / 42522 / 42522 / 42042（旧服务已释放；该数值为新服务加载过程中）。旧服务 `pid=55964` 已不存在，18010 在 08:05:54 后不再监听（`curl /health` exit 7）。

