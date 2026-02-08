from typing import Union, Literal
from pandas import DataFrame
from sqlalchemy import create_engine, text, exc
from config import HOST, USER, PASSWORD
from dtypes import users_dtypes, games_info_dtypes, games_by_moves_dtypes



def check_database_exists(username: str) -> Union[Literal['blitz', 'rapid'], None]:
    try:
        temp_db_url = f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}/postgres'
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
    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.db_url = f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}/postgres'
        self.engine = create_engine(self.db_url, echo=False)

    @staticmethod
    def check_database_exists(username: str) -> Union[Literal['blitz', 'rapid'], None]:
        try:
            temp_db_url = f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}/postgres'
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
                df_users.to_sql('users', con=connection, if_exists='replace',
                                      index=False, dtype=users_dtypes,
                                      schema=self.schema_name)
                games_info.to_sql('games_info', con=connection, if_exists='replace',
                                        index=False, dtype=games_info_dtypes,
                                        schema=self.schema_name)
                games_by_moves.to_sql('games_by_moves', con=connection, if_exists='replace',
                                      index=False, dtype=games_by_moves_dtypes,
                                      chunksize=100000, method='multi',
                                      schema=self.schema_name)
                self._add_primary_keys(connection)
                print(f"Data successfully saved to schema '{self.schema_name}'.")
        except exc.SQLAlchemyError as e:
            print(f'Error while saving data: {e}')

    def _add_primary_keys(self, connection):
        """Adding primary keys to tables"""
        try:
            connection.execute(text(f"""
                   ALTER TABLE {self.schema_name}.users 
                   ADD PRIMARY KEY (username);
               """))

            connection.execute(text(f"""
                   ALTER TABLE {self.schema_name}.games_info 
                   ADD PRIMARY KEY (id_game, id_player);
               """))

            connection.execute(text(f"""
                   ALTER TABLE {self.schema_name}.games_by_moves 
                   ADD PRIMARY KEY (id_game, move_number, main_color);
               """))

        except exc.SQLAlchemyError as e:
            print(f'Error while adding primary keys: {e}')

    def run_sql_script(self, script_path: str):
        try:
            with self.engine.connect() as connection:
                with open(script_path, 'r', encoding='utf-8') as sql_file:
                    sql_script = sql_file.read()
                    connection.execute(text(f"SET search_path TO {self.schema_name}"))
                    connection.execute(text(sql_script))
                connection.commit()
            print("SQL script executed successfully.")
        except Exception as e:
            print(f'Error while executing SQL script: {e}')

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