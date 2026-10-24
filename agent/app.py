"""On-instance monitoring agent.

Runs ON the GCP instance (in a container). Exposes read-only Docker and GPU
status as JSON, protected by a bearer token. The remote dashboard polls this.

Endpoints:
  GET /healthz        -> liveness, no auth
  GET /api/docker     -> docker daemon + container status (auth)
  GET /api/gpu        -> nvidia-smi GPU status + compute apps (auth)
  GET /api/all        -> combined snapshot (auth)
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status

try:
    import docker
    from docker.errors import DockerException
except ImportError:  # pragma: no cover
    docker = None
    DockerException = Exception

AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://localhost:8000/metrics")

app = FastAPI(title="vLLM Monitor Agent", version="1.0.0")

# A single shared Docker client, created lazily so the agent can still start
# (and report the error) if the socket is missing.
_docker_client = None


def get_docker_client():
    global _docker_client
    if _docker_client is None and docker is not None:
        _docker_client = docker.from_env()
    return _docker_client


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Constant-time bearer token check."""
    if not AGENT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AGENT_TOKEN not configured on the agent.",
        )
    expected = f"Bearer {AGENT_TOKEN}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_docker_time(value: str | None) -> datetime | None:
    if not value or value.startswith("0001-01-01"):
        return None
    # Docker timestamps look like 2024-01-02T03:04:05.123456789Z; python's
    # fromisoformat can't handle nanoseconds, so trim to microseconds.
    cleaned = value.replace("Z", "+00:00")
    if "." in cleaned:
        head, _, tail = cleaned.partition(".")
        frac = tail
        tz = ""
        for marker in ("+", "-"):
            if marker in frac:
                frac, _, tz = frac.partition(marker)
                tz = marker + tz
                break
        frac = frac[:6]
        cleaned = f"{head}.{frac}{tz}"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _container_stats(container) -> dict[str, Any] | None:
    """One-shot resource usage for a running container (like `docker stats`)."""
    try:
        s = container.stats(stream=False)
        cpu, pre = s.get("cpu_stats", {}), s.get("precpu_stats", {})
        cpu_delta = cpu.get("cpu_usage", {}).get("total_usage", 0) - pre.get("cpu_usage", {}).get("total_usage", 0)
        sys_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
        ncpu = cpu.get("online_cpus") or len(cpu.get("cpu_usage", {}).get("percpu_usage") or []) or 1
        cpu_pct = round(cpu_delta / sys_delta * ncpu * 100, 1) if sys_delta > 0 else None

        mem = s.get("memory_stats", {})
        usage = mem.get("usage")
        mstats = mem.get("stats", {})
        cache = mstats.get("inactive_file", mstats.get("cache", 0))
        mem_used = (usage - cache) if usage is not None else None
        limit = mem.get("limit")
        mem_pct = round(mem_used / limit * 100, 1) if mem_used is not None and limit else None

        rx = tx = 0
        for n in (s.get("networks") or {}).values():
            rx += n.get("rx_bytes", 0)
            tx += n.get("tx_bytes", 0)
        return {
            "cpu_pct": cpu_pct,
            "mem_used": mem_used,
            "mem_limit": limit,
            "mem_pct": mem_pct,
            "net_rx": rx,
            "net_tx": tx,
        }
    except Exception:  # noqa: BLE001
        return None


def collect_docker() -> dict[str, Any]:
    if docker is None:
        return {"ok": False, "error": "docker SDK not available"}
    try:
        # from_env() itself raises DockerException if the socket is missing,
        # so client creation must live inside the try.
        client = get_docker_client()
        info = client.info()
        containers = []
        for c in client.containers.list(all=True):
            attrs = c.attrs
            state = attrs.get("State", {})
            health = state.get("Health", {}).get("Status")  # None if no healthcheck
            started = _parse_docker_time(state.get("StartedAt"))
            uptime_s = None
            if started and state.get("Running"):
                uptime_s = int((datetime.now(timezone.utc) - started).total_seconds())
            entry = {
                "id": c.short_id,
                "name": c.name,
                "image": (c.image.tags[0] if c.image.tags else c.image.short_id),
                "status": c.status,  # running / exited / ...
                "health": health,  # healthy / unhealthy / starting / None
                "started_at": state.get("StartedAt"),
                "uptime_seconds": uptime_s,
                "restart_count": attrs.get("RestartCount", 0),
                "exit_code": state.get("ExitCode"),
                "ports": attrs.get("NetworkSettings", {}).get("Ports", {}),
            }
            if state.get("Running"):
                entry["stats"] = _container_stats(c)
            containers.append(entry)
        containers.sort(key=lambda x: (x["status"] != "running", x["name"]))
        return {
            "ok": True,
            "daemon": {
                "containers_running": info.get("ContainersRunning"),
                "containers_total": info.get("Containers"),
                "images": info.get("Images"),
                "server_version": info.get("ServerVersion"),
                "driver": info.get("Driver"),
                "ncpu": info.get("NCPU"),
                "mem_total": info.get("MemTotal"),
            },
            "containers": containers,
        }
    except DockerException as exc:
        return {"ok": False, "error": f"docker error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


_GPU_FIELDS = [
    "index",
    "name",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "fan.speed",
    "clocks.sm",
    "clocks.mem",
    # Health / link extras.
    "driver_version",
    "pstate",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "ecc.errors.uncorrected.volatile.total",
]

# Throttle/clock-event reasons queried separately so a naming mismatch on some
# driver versions can't break the main GPU query. Reported as active flags.
_GPU_THROTTLE_FIELDS = [
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
]


def _to_num(value: str) -> float | int | str | None:
    value = value.strip()
    if value in ("", "[N/A]", "[Not Supported]", "N/A"):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def collect_gpu() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {
            "ok": False,
            "error": "nvidia-smi not found (is the NVIDIA runtime / --gpus set?)",
        }
    try:
        query = ",".join(_GPU_FIELDS)
        out = subprocess.run(
            [smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(_GPU_FIELDS):
                continue
            row = {field: _to_num(val) for field, val in zip(_GPU_FIELDS, parts)}
            # name/uuid should stay strings
            row["name"] = parts[_GPU_FIELDS.index("name")]
            row["uuid"] = parts[_GPU_FIELDS.index("uuid")]
            mem_total = row.get("memory.total")
            mem_used = row.get("memory.used")
            if isinstance(mem_total, (int, float)) and mem_total and isinstance(mem_used, (int, float)):
                row["memory.used_pct"] = round(mem_used / mem_total * 100, 1)
            gpus.append(row)

        # Throttle reasons (best-effort; skipped if the driver uses other names).
        try:
            tq = ",".join(_GPU_THROTTLE_FIELDS)
            t_out = subprocess.run(
                [smi, f"--query-gpu={tq}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            for gpu_row, line in zip(gpus, t_out.stdout.strip().splitlines()):
                vals = [p.strip() for p in line.split(",")]
                active = [
                    field.split(".")[-1]
                    for field, val in zip(_GPU_THROTTLE_FIELDS, vals)
                    if val.lower() == "active"
                ]
                gpu_row["throttling"] = active
                gpu_row["throttled"] = bool(active)
        except Exception:  # noqa: BLE001
            pass

        apps = []
        try:
            app_out = subprocess.run(
                [
                    smi,
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            for line in app_out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    apps.append(
                        {
                            "pid": _to_num(parts[0]),
                            "process_name": parts[1],
                            "used_memory_mib": _to_num(parts[2]),
                        }
                    )
        except subprocess.SubprocessError:
            pass  # compute-apps can fail on some setups; GPU stats still useful

        return {"ok": True, "gpus": gpus, "compute_apps": apps}
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "error": f"nvidia-smi failed: {exc.stderr.strip()}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "nvidia-smi timed out"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _parse_prometheus(text: str) -> dict[str, list[tuple[str, float]]]:
    """Parse Prometheus exposition text into {metric_name: [(labels, value)]}."""
    out: dict[str, list[tuple[str, float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # name{labels} value   OR   name value
        if "{" in line:
            name, rest = line.split("{", 1)
            labels, _, value = rest.rpartition("}")
            value = value.strip()
        else:
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name, value = parts
            labels = ""
        try:
            val = float(value)
        except ValueError:
            continue
        out.setdefault(name.strip(), []).append((labels, val))
    return out


def _resolve(parsed: dict, name: str):
    """Look up a metric by canonical vLLM name, tolerating backends (e.g. NVIDIA
    NIM) that emit the same metrics WITHOUT the `vllm:` prefix."""
    if name in parsed:
        return parsed[name]
    if name.startswith("vllm:"):
        bare = name[len("vllm:"):]
        if bare in parsed:
            return parsed[bare]
    return None


def _sum_metric(parsed: dict, name: str) -> float | None:
    series = _resolve(parsed, name)
    if not series:
        return None
    return sum(v for _, v in series)


def _histogram_quantile(parsed: dict, name: str, q: float) -> float | None:
    """Estimate a quantile from Prometheus histogram buckets (like PromQL
    histogram_quantile). Aggregates bucket counts across label sets so it works
    with multiple engines/models."""
    series = _resolve(parsed, name + "_bucket")
    if not series:
        return None
    agg: dict[float, float] = {}
    for labels, val in series:
        if 'le="' not in labels:
            continue
        le_raw = labels.split('le="', 1)[1].split('"', 1)[0]
        le = float("inf") if le_raw in ("+Inf", "Inf") else float(le_raw)
        agg[le] = agg.get(le, 0.0) + val
    if not agg:
        return None
    buckets = sorted(agg.items())  # [(le, cumulative_count)]
    total = buckets[-1][1]  # +Inf bucket holds the full count
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le  # open-ended top bucket; best estimate is its lower edge
            if count == prev_count:
                return le
            frac = (rank - prev_count) / (count - prev_count)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_count = le, count
    return buckets[-1][0]


# Previous sample for computing token throughput (rate) between scrapes.
_vllm_prev: dict[str, float] = {}


def _vllm_metric_urls() -> list[str]:
    """Configured URL plus the alternate path, so we auto-detect both vLLM
    (`/metrics`) and NVIDIA NIM (`/v1/metrics`) without reconfiguration."""
    urls = [VLLM_METRICS_URL]
    if VLLM_METRICS_URL.endswith("/v1/metrics"):
        urls.append(VLLM_METRICS_URL[: -len("/v1/metrics")] + "/metrics")
    elif VLLM_METRICS_URL.endswith("/metrics"):
        urls.append(VLLM_METRICS_URL[: -len("/metrics")] + "/v1/metrics")
    return urls


def collect_vllm() -> dict[str, Any]:
    parsed = None
    last_err = None
    for url in _vllm_metric_urls():
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            last_err = f"could not scrape {url}: {exc}"
            continue
        candidate = _parse_prometheus(text)
        # Accept the endpoint only if it actually carries serving metrics
        # (prefix-agnostic), so a 404 body or wrong port is rejected.
        if _resolve(candidate, "vllm:num_requests_running") is not None or \
           _resolve(candidate, "vllm:generation_tokens_total") is not None:
            parsed = candidate
            break
        last_err = f"no vLLM/NIM serving metrics at {url}"

    if parsed is None:
        return {"ok": False, "error": last_err or "no metrics endpoint reachable"}

    # Model name from any series that carries the label.
    model = None
    for series in parsed.values():
        for labels, _ in series:
            if "model_name=" in labels:
                model = labels.split('model_name="', 1)[1].split('"', 1)[0]
                break
        if model:
            break

    prompt_total = _sum_metric(parsed, "vllm:prompt_tokens_total")
    gen_total = _sum_metric(parsed, "vllm:generation_tokens_total")

    # Token throughput: delta since last scrape.
    now = time.monotonic()
    prompt_tps = gen_tps = None
    if _vllm_prev:
        dt = now - _vllm_prev.get("t", now)
        if dt > 0:
            if prompt_total is not None and "prompt" in _vllm_prev:
                prompt_tps = max(0.0, (prompt_total - _vllm_prev["prompt"]) / dt)
            if gen_total is not None and "gen" in _vllm_prev:
                gen_tps = max(0.0, (gen_total - _vllm_prev["gen"]) / dt)
    _vllm_prev.update({"t": now})
    if prompt_total is not None:
        _vllm_prev["prompt"] = prompt_total
    if gen_total is not None:
        _vllm_prev["gen"] = gen_total

    def avg(sum_name: str, count_name: str, scale: float = 1.0, nd: int = 1) -> float | None:
        s = _sum_metric(parsed, sum_name)
        c = _sum_metric(parsed, count_name)
        if s is None or not c:
            return None
        return round(s / c * scale, nd)

    def pct(hist: str, q: float, nd: int = 3) -> float | None:
        v = _histogram_quantile(parsed, hist, q)
        return round(v, nd) if v is not None else None

    # vLLM: kv_cache_usage_perc; NVIDIA NIM: gpu_cache_usage_perc.
    kv = _sum_metric(parsed, "vllm:kv_cache_usage_perc")
    if kv is None:
        kv = _sum_metric(parsed, "gpu_cache_usage_perc")

    # Prefix-cache hit rate (only meaningful when prefix caching is enabled).
    pc_hits = _sum_metric(parsed, "vllm:prefix_cache_hits_total")
    pc_queries = _sum_metric(parsed, "vllm:prefix_cache_queries_total")
    prefix_hit_rate = (
        round(pc_hits / pc_queries * 100, 1)
        if pc_hits is not None and pc_queries else None
    )

    e2e = "vllm:e2e_request_latency_seconds"
    return {
        "ok": True,
        "model": model,
        "requests_running": _sum_metric(parsed, "vllm:num_requests_running"),
        "requests_waiting": _sum_metric(parsed, "vllm:num_requests_waiting"),
        # vLLM reports kv cache as a 0..1 fraction; expose as percent.
        "kv_cache_usage_pct": round(kv * 100, 1) if kv is not None else None,
        # Throughput (rate between scrapes).
        "input_tokens_per_sec": round(prompt_tps, 1) if prompt_tps is not None else None,
        "output_tokens_per_sec": round(gen_tps, 1) if gen_tps is not None else None,
        # Latency.
        "avg_ttft_ms": avg("vllm:time_to_first_token_seconds_sum",
                           "vllm:time_to_first_token_seconds_count", scale=1000),
        "avg_inter_token_ms": avg("vllm:inter_token_latency_seconds_sum",
                                  "vllm:inter_token_latency_seconds_count", scale=1000),
        "latency_p50_s": pct(e2e, 0.50),
        "latency_p95_s": pct(e2e, 0.95),
        "latency_p99_s": pct(e2e, 0.99),
        "ttft_p95_ms": (lambda v: round(v * 1000, 1) if v is not None else None)(
            _histogram_quantile(parsed, "vllm:time_to_first_token_seconds", 0.95)),
        # Cache pressure.
        "preemptions_total": _sum_metric(parsed, "vllm:num_preemptions_total"),
        "prefix_cache_hit_rate_pct": prefix_hit_rate,
        # Token counts.
        "prompt_tokens_total": prompt_total,
        "generation_tokens_total": gen_total,
        "avg_prompt_tokens_per_req": avg("vllm:request_prompt_tokens_sum",
                                         "vllm:request_prompt_tokens_count", nd=0),
        "avg_output_tokens_per_req": avg("vllm:request_generation_tokens_sum",
                                         "vllm:request_generation_tokens_count", nd=0),
        "requests_finished": _sum_metric(parsed, "vllm:request_success_total"),
    }


VLLM_CONTAINER = os.environ.get("VLLM_CONTAINER", "vllm")


def collect_vllm_config() -> dict[str, Any]:
    """Inspect the vLLM container to surface how it was launched: engine args
    plus runtime flags (image, gpus, shm, ports, volumes). Deliberately does
    NOT return env vars, which may contain secrets (e.g. HF_TOKEN)."""
    if docker is None:
        return {"ok": False, "error": "docker SDK not available"}
    try:
        client = get_docker_client()
        # Known inference-server markers (vLLM, NVIDIA NIM, TensorRT-LLM, TGI, SGLang).
        patterns = ("vllm", "nim", "nvcr.io/nim", "tensorrt", "trt-llm", "tgi", "sglang")

        def looks_like_server(c) -> bool:
            if "monitor" in c.name.lower():
                return False  # never match our own agent
            img = (c.image.tags[0] if c.image.tags else "").lower()
            return any(p in img or p in c.name.lower() for p in patterns)

        all_containers = client.containers.list(all=True)
        container = None
        # Prefer a *running* server (so a stopped vllm doesn't shadow a live nim).
        running = [c for c in all_containers if c.status == "running" and looks_like_server(c)]
        if running:
            container = running[0]
        else:
            try:
                container = client.containers.get(VLLM_CONTAINER)
            except Exception:  # noqa: BLE001
                matches = [c for c in all_containers if looks_like_server(c)]
                container = matches[0] if matches else None
        if container is None:
            return {"ok": False, "error": f"no inference container found (looked for {VLLM_CONTAINER} / nim / vllm / …)"}

        attrs = container.attrs
        cfg = attrs.get("Config", {})
        host = attrs.get("HostConfig", {})

        # Parse engine args from entrypoint + cmd into key/value pairs.
        tokens = list(cfg.get("Entrypoint") or []) + list(cfg.get("Cmd") or [])
        engine_args: dict[str, Any] = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if isinstance(tok, str) and tok.startswith("--"):
                key = tok[2:]
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                if isinstance(nxt, str) and not nxt.startswith("--"):
                    engine_args[key] = nxt
                    i += 2
                else:
                    engine_args[key] = True  # boolean flag
                    i += 1
            else:
                i += 1

        # GPU request.
        gpus = None
        for dr in host.get("DeviceRequests") or []:
            caps = [c for grp in (dr.get("Capabilities") or []) for c in grp]
            if "gpu" in caps:
                if dr.get("DeviceIDs"):
                    gpus = ",".join(dr["DeviceIDs"])
                elif dr.get("Count") in (-1, None):
                    gpus = "all"
                else:
                    gpus = str(dr["Count"])

        # Port bindings -> "127.0.0.1:8000 -> 8000/tcp".
        ports = []
        for cport, binds in (host.get("PortBindings") or {}).items():
            for b in binds or []:
                hip = b.get("HostIp") or "0.0.0.0"
                ports.append(f"{hip}:{b.get('HostPort')} -> {cport}")

        # Mounts/volumes.
        mounts = []
        for m in attrs.get("Mounts") or []:
            src = m.get("Name") or m.get("Source")
            mounts.append({"source": src, "dest": m.get("Destination"), "type": m.get("Type")})

        # Identify the serving engine from the container image/name so the UI
        # can label itself (NIM vs vLLM vs …) without a manual switch.
        img_l = (cfg.get("Image") or "").lower()
        name_l = container.name.lower()
        if "nvcr.io/nim" in img_l or "nim" in name_l:
            engine = "NIM"
        elif "vllm" in img_l or "vllm" in name_l:
            engine = "vLLM"
        elif "tensorrt" in img_l or "trt-llm" in img_l or "triton" in img_l:
            engine = "TensorRT-LLM"
        elif "text-generation-inference" in img_l or "/tgi" in img_l:
            engine = "TGI"
        elif "sglang" in img_l:
            engine = "SGLang"
        else:
            engine = "LLM"

        return {
            "ok": True,
            "engine": engine,
            "container": container.name,
            "image": cfg.get("Image"),
            "runtime": host.get("Runtime"),
            "gpus": gpus,
            "shm_size_bytes": host.get("ShmSize"),
            "restart_policy": (host.get("RestartPolicy") or {}).get("Name"),
            "ports": ports,
            "mounts": mounts,
            "engine_args": engine_args,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# Previous samples for host CPU% and network rate.
_host_prev: dict[str, Any] = {}
# Host root filesystem is bind-mounted here (read-only) by the compose file.
HOST_ROOT = "/host"


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def collect_host() -> dict[str, Any]:
    """System-level metrics read from /proc (host-visible via pid:host) plus
    disk usage from the read-only host mount and Docker volume sizes."""
    out: dict[str, Any] = {"ok": True}
    now = time.monotonic()

    # Load average.
    try:
        la = _read_file("/proc/loadavg").split()
        out["load1"], out["load5"], out["load15"] = float(la[0]), float(la[1]), float(la[2])
    except Exception:  # noqa: BLE001
        pass

    # CPU utilization (delta of /proc/stat between scrapes).
    try:
        parts = [int(x) for x in _read_file("/proc/stat").splitlines()[0].split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle + iowait
        total = sum(parts)
        prev = _host_prev.get("cpu")
        if prev:
            dt_total, dt_idle = total - prev[0], idle - prev[1]
            if dt_total > 0:
                out["cpu_pct"] = round((1 - dt_idle / dt_total) * 100, 1)
        _host_prev["cpu"] = (total, idle)
    except Exception:  # noqa: BLE001
        pass

    # Memory + swap.
    try:
        mem: dict[str, int] = {}
        for line in _read_file("/proc/meminfo").splitlines():
            k, _, v = line.partition(":")
            mem[k.strip()] = int(v.strip().split()[0]) * 1024  # kB -> bytes
        total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
        out["mem_total"] = total
        if total and avail is not None:
            out["mem_used"] = total - avail
            out["mem_used_pct"] = round((total - avail) / total * 100, 1)
        st, sf = mem.get("SwapTotal"), mem.get("SwapFree")
        if st:
            out["swap_total"] = st
            out["swap_used"] = st - sf if sf is not None else None
            out["swap_used_pct"] = round((st - sf) / st * 100, 1) if sf is not None else None
    except Exception:  # noqa: BLE001
        pass

    # Uptime + process count.
    try:
        out["uptime_seconds"] = int(float(_read_file("/proc/uptime").split()[0]))
    except Exception:  # noqa: BLE001
        pass
    try:
        out["processes"] = sum(1 for n in os.listdir("/proc") if n.isdigit())
    except Exception:  # noqa: BLE001
        pass

    # Network throughput (sum of non-loopback interfaces).
    try:
        rx = tx = 0
        for line in _read_file("/proc/net/dev").splitlines():
            if ":" not in line:
                continue
            iface, _, data = line.partition(":")
            if iface.strip() == "lo":
                continue
            f = data.split()
            rx += int(f[0])
            tx += int(f[8])
        out["net_rx_bytes"], out["net_tx_bytes"] = rx, tx
        prev = _host_prev.get("net")
        if prev:
            dt = now - prev[2]
            if dt > 0:
                out["net_rx_bytes_per_sec"] = round(max(0.0, (rx - prev[0]) / dt))
                out["net_tx_bytes_per_sec"] = round(max(0.0, (tx - prev[1]) / dt))
        _host_prev["net"] = (rx, tx, now)
    except Exception:  # noqa: BLE001
        pass

    # Disk usage of the host root (requires the /host read-only mount).
    try:
        s = os.statvfs(HOST_ROOT)
        total = s.f_blocks * s.f_frsize
        free = s.f_bavail * s.f_frsize
        used = total - free
        out["disk"] = {
            "mount": "/",
            "total": total,
            "used": used,
            "free": free,
            "used_pct": round(used / total * 100, 1) if total else None,
        }
    except Exception:  # noqa: BLE001
        pass

    # Docker volume sizes (e.g. the HuggingFace model cache).
    try:
        if docker is not None:
            df = get_docker_client().df()
            vols = []
            for v in df.get("Volumes") or []:
                size = (v.get("UsageData") or {}).get("Size")
                if size and size > 0:
                    vols.append({"name": v.get("Name"), "size": size})
            if vols:
                out["volumes"] = sorted(vols, key=lambda x: -x["size"])[:5]
    except Exception:  # noqa: BLE001
        pass

    return out


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "time": _now_iso()}


@app.get("/api/docker", dependencies=[Depends(require_token)])
def api_docker() -> dict[str, Any]:
    return {"time": _now_iso(), "docker": collect_docker()}


@app.get("/api/gpu", dependencies=[Depends(require_token)])
def api_gpu() -> dict[str, Any]:
    return {"time": _now_iso(), "gpu": collect_gpu()}


@app.get("/api/vllm", dependencies=[Depends(require_token)])
def api_vllm() -> dict[str, Any]:
    return {"time": _now_iso(), "vllm": collect_vllm()}


@app.get("/api/vllm-config", dependencies=[Depends(require_token)])
def api_vllm_config() -> dict[str, Any]:
    return {"time": _now_iso(), "vllm_config": collect_vllm_config()}


@app.get("/api/host", dependencies=[Depends(require_token)])
def api_host() -> dict[str, Any]:
    return {"time": _now_iso(), "host": collect_host()}


@app.get("/api/all", dependencies=[Depends(require_token)])
def api_all() -> dict[str, Any]:
    return {
        "time": _now_iso(),
        "host": collect_host(),
        "docker": collect_docker(),
        "gpu": collect_gpu(),
        "vllm": collect_vllm(),
        "vllm_config": collect_vllm_config(),
    }
