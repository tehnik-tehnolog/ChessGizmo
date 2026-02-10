from typing import Union, Literal
from pandas import DataFrame
from sqlalchemy import create_engine, text, exc
from importlib import resources
from typing import Optional
from .dtypes import users_dtypes, games_info_dtypes, games_by_moves_dtypes
from .config import GizmoConfig


def check_database_exists(username: str, config: Optional[GizmoConfig] = None) -> Union[Literal['blitz', 'rapid'], None]:
    try:
        cfg = config or GizmoConfig.from_env()
        temp_db_url = f'postgresql+psycopg2://{cfg.user}:{cfg.password}@{cfg.host}/postgres'
        temp_engine = create_engine(temp_db_url, echo=False)
        with temp_engine.connect() as connection:
            blitz_schema = f'chess_blitz_{username}'
            rapid_schema = f'chess_rapid_{username}'

            query = text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name = :schema_name
            """)

            if connection.execute(query, {"schema_name": blitz_schema}).fetchone() is not None:
                return 'blitz'
            elif connection.execute(query, {"schema_name": rapid_schema}).fetchone() is not None:
                return 'rapid'
    except exc.SQLAlchemyError as e:
        print(f'Error checking schema existence: {e}')
        return None


class PopulateDB:
    def __init__(self, schema_name: str, config: Optional[GizmoConfig] = None):
        # Санитайзинг имени схемы (только буквы, цифры и подчеркивание)
        self.schema_name = ''.join(c for c in schema_name if c.isalnum() or c == '_')
        cfg = config or GizmoConfig.from_env()
        self.db_url = f'postgresql+psycopg2://{cfg.user}:{cfg.password}@{cfg.host}/postgres'
        self.engine = create_engine(self.db_url, echo=False)

    @staticmethod
    def check_database_exists(username: str, config: Optional[GizmoConfig] = None) -> Union[Literal['blitz', 'rapid'], None]:
        try:
            cfg = config or GizmoConfig.from_env()
            temp_db_url = f'postgresql+psycopg2://{cfg.user}:{cfg.password}@{cfg.host}/postgres'
            temp_engine = create_engine(temp_db_url, echo=False)
            with temp_engine.connect() as connection:
                blitz_schema = f'chess_blitz_{username}'
                rapid_schema = f'chess_rapid_{username}'

                query = text("""
                    SELECT schema_name 
                    FROM information_schema.schemata 
                    WHERE schema_name = :schema_name
                """)

                if connection.execute(query, {"schema_name": blitz_schema}).fetchone() is not None:
                    return 'blitz'
                elif connection.execute(query, {"schema_name": rapid_schema}).fetchone() is not None:
                    return 'rapid'
        except exc.SQLAlchemyError as e:
            print(f'Error checking schema existence: {e}')
            return None

    def create_database(self):
        try:
            with self.engine.connect() as connection:
                connection.execute(text(f"CREATE SCHEMA {self.schema_name}"))
                connection.commit()
                print(f"Schema '{self.schema_name}' successfully created.")
        except exc.SQLAlchemyError as e:
            if "already exists" in str(e):
                print(f"Schema '{self.schema_name}' already exists.")
            else:
                print(f'Error while creating schema: {e}')

    def drop_database(self):
        try:
            self.engine.dispose()
            temp_engine = create_engine(
                self.db_url,
                echo=False,
                isolation_level="AUTOCOMMIT"
            )

            with temp_engine.connect() as connection:
                connection.execute(text(f"DROP SCHEMA IF EXISTS {self.schema_name} CASCADE"))
            print(f"Schema '{self.schema_name}' successfully deleted.")
        except exc.SQLAlchemyError as e:
            print(f'Error while deleting schema: {e}')

    def save_df(self, df_users: DataFrame, games_info: DataFrame, games_by_moves: DataFrame):
        try:
            with self.engine.begin() as connection:
                # Установка search_path для текущей транзакции
                connection.execute(text(f"SET search_path TO {self.schema_name}"))

                # Запись таблиц
                tables = [
                    (df_users, 'users', users_dtypes),
                    (games_info, 'games_info', games_info_dtypes),
                    (games_by_moves, 'games_by_moves', games_by_moves_dtypes)
                ]

                for df, name, dtypes in tables:
                    # chunksize для всех таблиц для стабильности
                    df.to_sql(name, con=connection, if_exists='replace', index=False,
                              dtype=dtypes, schema=self.schema_name, chunksize=5000)

                self._add_primary_keys(connection)
        except exc.SQLAlchemyError as e:
            print(f"Save error: {e}")

    def _add_primary_keys(self, connection):
        """Adding primary keys to tables"""
        queries = [
            f"ALTER TABLE {self.schema_name}.users ADD PRIMARY KEY (username);",
            f"ALTER TABLE {self.schema_name}.games_info ADD PRIMARY KEY (id_game, id_player);",
            f"ALTER TABLE {self.schema_name}.games_by_moves ADD PRIMARY KEY (id_game, move_number, main_color);"
        ]
        for q in queries:
            connection.execute(text(q))

    def run_sql_script(self, script_name: str):
        """
                Выполняет SQL скрипт, хранящийся внутри пакета.
                param script_name: Имя файла .sql'
        """
        try:
            script_content = resources.files('chessgizmo.sql.PostgreSQL_scripts').joinpath(script_name).read_text(
                encoding='utf-8')

            with self.engine.begin() as connection:
                connection.execute(text(f"SET search_path TO {self.schema_name}"))
                connection.execute(text(script_content))
            print(f"Script {script_name} executed.")
        except Exception as e:
            print(f"Error executing script {script_name}: {e}")

    def get_dataframe(self, query: str) -> Union[DataFrame, None]:
        try:
            with self.engine.connect() as connection:
                connection.execute(text(f"SET search_path TO {self.schema_name}"))
                result = connection.execute(text(query))
                df = DataFrame(result.fetchall(), columns=result.keys())
                return df
        except exc.SQLAlchemyError as e:
            print(f'Error while executing SQL query: {e}')
            return None