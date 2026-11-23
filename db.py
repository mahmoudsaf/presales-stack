import os
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "https://presales-db-mahmoudsaf.aws-eu-central-1.turso.io")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3OTU0MzgwNjksImlkIjoiMDFhMjA4NzItNjAwMS03ZmMyLThhMzQtMGExMDQ0NTM2M2Y2Iiwia2lkIjoiakQzTFBUZ2F5WGtCdEwxY0xTUVpnb0Y5VnhZWWd1Rkp3SlJ3dEp3UDc2YyIsInJpZCI6IjBmYmE5ZTM2LTQxMGItNDA0Mi1hNjZlLTQyN2Y0OTMyZjM4ZiJ9.1EooxDmPyNgtWOv-92rfaXTo6j7VIAdGNa_QR48mQgk0FyoQimSSDPGNdtMqSQ1YlOAdToXBTaEDZ8WbvqMsAQ")
USE_LOCAL = os.getenv("USE_LOCAL_SQLITE", "").lower() in ("1", "true", "yes")

BASE_DIR = Path(__file__).resolve().parent
CRM_DB_FILE = BASE_DIR / "crm.db"
TASKS_DB_FILE = BASE_DIR / "tasks.db"

try:
    import libsql_client
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False


class TursoRow(dict):
    """Row object behaving like sqlite3.Row: dict access, index access, and tuple unpacking."""
    def __init__(self, cols: Tuple[str, ...], values: Tuple[Any, ...]):
        super().__init__(zip(cols, values))
        self._values = tuple(values)
        self._cols = tuple(cols)

    def __getitem__(self, item: Union[int, str]) -> Any:
        if isinstance(item, int):
            return self._values[item]
        return super().__getitem__(item)

    def __iter__(self):
        return iter(self._values)


class TursoCursor:
    def __init__(self, client: Any):
        self.client = client
        self.last_res = None
        self._row_idx = 0
        self._converted_rows: List[TursoRow] = []

    def execute(self, sql: str, params: Optional[Union[List[Any], Tuple[Any, ...]]] = None):
        if params is None:
            p_list = []
        elif isinstance(params, (tuple, list)):
            p_list = list(params)
        else:
            p_list = [params]

        try:
            self.last_res = self.client.execute(sql, p_list)
        except Exception as e:
            print(f"[Turso Exec Error] SQL: {sql}")
            print(f"[Turso Exec Error] Params: {p_list}")
            raise e
        self._row_idx = 0
        cols = tuple(self.last_res.columns)
        self._converted_rows = [TursoRow(cols, tuple(row)) for row in self.last_res.rows]
        return self

    def executemany(self, sql: str, seq_of_params: Any):
        for params in seq_of_params:
            self.execute(sql, params)
        return self

    def fetchone(self) -> Optional[TursoRow]:
        if not self._converted_rows or self._row_idx >= len(self._converted_rows):
            return None
        row = self._converted_rows[self._row_idx]
        self._row_idx += 1
        return row

    def fetchall(self) -> List[TursoRow]:
        if not self._converted_rows:
            return []
        rows = self._converted_rows[self._row_idx:]
        self._row_idx = len(self._converted_rows)
        return rows

    def __iter__(self):
        return iter(self._converted_rows)

    @property
    def lastrowid(self) -> Optional[int]:
        if self.last_res and self.last_res.last_insert_rowid is not None:
            try:
                return int(self.last_res.last_insert_rowid)
            except Exception:
                return None
        return None

    @property
    def rowcount(self) -> int:
        return self.last_res.rows_affected if self.last_res else 0

    @property
    def description(self) -> Optional[List[Tuple]]:
        if not self.last_res or not self.last_res.columns:
            return None
        return [(col, None, None, None, None, None, None) for col in self.last_res.columns]

    def close(self):
        pass


class TursoConnection:
    """Connection wrapper providing a sqlite3-compatible interface over Turso Cloud."""
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self._client = libsql_client.create_client_sync(url=self.url, auth_token=self.token)
        self.row_factory = None

    def cursor(self) -> TursoCursor:
        return TursoCursor(self._client)

    def execute(self, sql: str, params: Optional[Any] = None) -> TursoCursor:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self):
        # Turso pipeline executes statements committed per request
        pass

    def rollback(self):
        pass

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()


def get_db_connection(db_fallback_path: Optional[Path] = None):
    """
    Returns an active database connection:
    - If Turso is available and not disabled: returns a Turso cloud connection.
    - Otherwise: falls back to local SQLite file.
    """
    if HAS_LIBSQL and TURSO_DATABASE_URL and TURSO_AUTH_TOKEN and not USE_LOCAL:
        try:
            return TursoConnection(url=TURSO_DATABASE_URL, token=TURSO_AUTH_TOKEN)
        except Exception as e:
            print(f"[DB Warning] Turso connection failed ({e}), falling back to local SQLite.")

    # Fallback to local SQLite
    target_path = db_fallback_path or CRM_DB_FILE
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn
