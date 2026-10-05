"""Builds Nsight Systems-shaped SQLite files for tests: the subset of the exporter's schema servebench.nsys.parse
reads (column names as in the nsys exporter reference; tables exist only when given rows, as in real exports).

    path = fake_trace(tmp_path / "t.sqlite",
                      kernels=[(start, end, "name"), (start, end, "name", {"streamId": 7, "graphNodeId": 3})],
                      graphs=[(start, end, graph_id)], memcpys=[(start, end), (start, end, copy_kind, bytes)],
                      memsets=[(start, end)],
                      graph_table="CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
"""
import sqlite3


def fake_trace(path, kernels=(), graphs=(), memcpys=(), memsets=(), graph_table="CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
               analysis=None, session_start=None, metrics=None):
    """analysis=(start, stop) is written as ANALYSIS_DETAILS, on an absolute clock when session_start is given (as real
    exports do: activity rows count from TARGET_INFO_SESSION_START_TIME.systemClockNs). metrics={name: [(t, value)]}
    is written as GPU_METRICS with its names in TARGET_INFO_GPU_METRICS."""
    con = sqlite3.connect(path)
    strings = {}

    def sid(text):
        return strings.setdefault(text, len(strings))

    if kernels:
        con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, "
                    "streamId INTEGER, correlationId INTEGER, graphNodeId INTEGER, graphId INTEGER, "
                    "shortName INTEGER, demangledName INTEGER)")
        for i, k in enumerate(kernels):
            start, end, name, extra = (*k, {}) if len(k) == 3 else k
            row = {"start": start, "end": end, "deviceId": 0, "streamId": 7, "correlationId": i,
                   "graphNodeId": None, "graphId": None, "shortName": sid(name.split("<")[0]),
                   "demangledName": sid(name), **extra}
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (:start, :end, :deviceId, :streamId, "
                        ":correlationId, :graphNodeId, :graphId, :shortName, :demangledName)", row)
    for table, rows in (("CUPTI_ACTIVITY_KIND_MEMCPY", memcpys), ("CUPTI_ACTIVITY_KIND_MEMSET", memsets)):
        if rows:  # a copy is (start, end) or (start, end, copyKind, bytes): copyKind 2 is device to host
            con.execute(f"CREATE TABLE {table} (start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, "
                        "graphNodeId INTEGER, copyKind INTEGER, bytes INTEGER)")
            con.executemany(f"INSERT INTO {table} VALUES (?, ?, 0, 7, NULL, ?, ?)",
                            [(*r[:2], *(r[2:] or (1, 0))) for r in rows])
    if graphs:
        con.execute(f"CREATE TABLE {graph_table} (start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, "
                    "graphId INTEGER, correlationId INTEGER)")
        con.executemany(f"INSERT INTO {graph_table} VALUES (?, ?, 0, 7, ?, ?)",
                        [(s, e, g, 1000 + i) for i, (s, e, g) in enumerate(graphs)])
    if strings:
        con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        con.executemany("INSERT INTO StringIds VALUES (?, ?)", [(i, s) for s, i in strings.items()])
    if analysis:
        con.execute("CREATE TABLE ANALYSIS_DETAILS (startTime INTEGER, stopTime INTEGER, duration INTEGER)")
        con.execute("INSERT INTO ANALYSIS_DETAILS VALUES (?, ?, ?)", (*analysis, analysis[1] - analysis[0]))
    if session_start is not None:
        con.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, systemClockNs INTEGER)")
        con.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (0, ?)", (session_start,))
    if metrics:
        con.execute("CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, sourceId INTEGER, typeName TEXT, "
                    "samplingFrequency INTEGER, metricId INTEGER, metricName TEXT)")
        con.execute("CREATE TABLE GPU_METRICS (typeId INTEGER, timestamp INTEGER, value INTEGER, metricId INTEGER)")
        for mid, (name, samples) in enumerate(metrics.items()):
            con.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (1, 0, 'GH100', 10000, ?, ?)", (mid, name))
            con.executemany("INSERT INTO GPU_METRICS VALUES (1, ?, ?, ?)", [(t, v, mid) for t, v in samples])
    con.commit()
    con.close()
    return path
