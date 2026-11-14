"""Acesso ao SQLite do histórico.

O esquema original é preservado; as colunas `origem` e `movimento_real` são
acrescentadas automaticamente em bancos antigos (os registros existentes são
mantidos, com esses campos vazios).
"""
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

from . import config

ORIGEM_MANUAL = "manual"
ORIGEM_AMOSTRA_TESTE = "amostra_teste"

_COLUNAS_NOVAS = {"origem": "TEXT", "movimento_real": "TEXT"}


def _conectar() -> sqlite3.Connection:
    return sqlite3.connect(config.CAMINHO_BANCO)


def init_db() -> None:
    with closing(_conectar()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predicoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                acc_x REAL,
                acc_y REAL,
                acc_z REAL,
                gyro_x REAL,
                gyro_y REAL,
                gyro_z REAL,
                movimento_previsto TEXT
            )
            """
        )
        existentes = {linha[1] for linha in conn.execute("PRAGMA table_info(predicoes)")}
        for coluna, tipo in _COLUNAS_NOVAS.items():
            if coluna not in existentes:
                conn.execute(f"ALTER TABLE predicoes ADD COLUMN {coluna} {tipo}")
        conn.commit()


def salvar_predicao(
    dados: dict,
    movimento: str,
    origem: str = ORIGEM_MANUAL,
    movimento_real: str | None = None,
) -> None:
    with closing(_conectar()) as conn:
        conn.execute(
            """
            INSERT INTO predicoes
                (data_hora, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                 movimento_previsto, origem, movimento_real)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                float(dados["ACC_X"]), float(dados["ACC_Y"]), float(dados["ACC_Z"]),
                float(dados["GYRO_X"]), float(dados["GYRO_Y"]), float(dados["GYRO_Z"]),
                str(movimento), origem, movimento_real,
            ),
        )
        conn.commit()


def ler_historico() -> pd.DataFrame:
    with closing(_conectar()) as conn:
        return pd.read_sql_query("SELECT * FROM predicoes ORDER BY id DESC", conn)
