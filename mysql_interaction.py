from typing import Union, Literal
from pandas import DataFrame
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine.url import make_url
from config import HOST, USER, PASSWORD
from dtypes import chess_df_users_dtypes, chess_games_info_dtypes, games_by_moves_dtypes


def check_database_exists(username: str) -> Union[Literal['blitz', 'rapid'], None]:
    """
        Проверяет, существует ли база данных на сервере MySQL.
        """
    try:
        temp_db_url = fr'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}'
        temp_engine = create_engine(temp_db_url, echo=False)
        with temp_engine.connect() as connection:
            blitz_db_name = f'chess_blitz_{username}'
            rapid_db_name = f'chess_rapid_{username}'
            query = text("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = :db_name")
            if connection.execute(query, {"db_name": blitz_db_name}).fetchone() is not None:
                return 'blitz'
            elif connection.execute(query, {"db_name": rapid_db_name}).fetchone() is not None:
                return 'rapid'
    except IndexError:
        print('База данных отсутствует')
        return None
  #  finally:
  #      connection.close()


class PopulateDB:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.db_url = fr'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/{self.db_name}'
        self.temp_db_url = fr'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/'
        self.engine = create_engine(self.db_url, echo=False)
        self.temp_engine = create_engine(self.temp_db_url, echo=False)

    def check_database_exists(username: str) -> Union[Literal['blitz', 'rapid'], None]:
        """
            Проверяет, существует ли база данных на сервере MySQL.
            """
        try:
            temp_db_url = fr'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}'
            temp_engine = create_engine(temp_db_url, echo=False)
            with temp_engine.connect() as connection:
                blitz_db_name = f'chess_blitz_{username}'
                rapid_db_name = f'chess_rapid_{username}'
                query = text("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = :db_name")
                if connection.execute(query, {"db_name": blitz_db_name}).fetchone() is not None:
                    return 'blitz'
                elif connection.execute(query, {"db_name": rapid_db_name}).fetchone() is not None:
                    return 'rapid'
        except IndexError:
            print('База данных отсутствует')
            return None

    def create_database(self):
        """
        Создаёт базу данных, если она не существует.
        """
        try:
            with self.temp_engine.connect() as connection:
                connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {self.db_name}"))
            print(f"База данных '{self.db_name}' успешно создана или уже существует.")
        except exc.SQLAlchemyError as e:
            print(f'Ошибка при создании базы данных: {e}')
 #       finally:
  #          connection.close()

    def drop_database(self):
        try:
            with self.temp_engine.connect() as connection:
                connection.execute(text(f"DROP DATABASE IF EXISTS {self.db_name}"))
            print(f"База данных '{self.db_name}' успешно удалена.")
        except exc.SQLAlchemyError as e:
            print(f'Ошибка при базы данных: {e}')
  #      finally:
   #         connection.close()

    def save_df(self, chess_df_users: DataFrame, chess_games_info: DataFrame, games_by_moves: DataFrame):
        """
        Сохраняет DataFrame в таблицы MySQL.
        """
        try:
            with self.engine.begin() as connection:
                # Сохранение данных в таблицы
                chess_df_users.to_sql('chess_df_users', con=self.engine, if_exists='replace', index=False,
                                      dtype=chess_df_users_dtypes)
                chess_games_info.to_sql('chess_games_info', con=self.engine, if_exists='replace', index=False,
                                        dtype=chess_games_info_dtypes)
                games_by_moves.to_sql('games_by_moves', con=connection, if_exists='replace',
                                      index=False, dtype=games_by_moves_dtypes,
                                      chunksize=100000, method='multi')

                print("Данные успешно сохранены в таблицы MySQL.")
        except exc.SQLAlchemyError as e:
            print(f'Ошибка при сохранении данных: {e}')

    def run_sql_script(self, script_path: str):
        """Выполняет SQL-скрипт из файла"""
        try:
            with self.engine.connect() as connection:
                with open(script_path, 'r', encoding='utf-8') as sql_file:
                    sql_script = sql_file.read()
                    connection.execute(text(sql_script))
                connection.commit()
            print("SQL-скрипт успешно выполнен.")
        except Exception as e:
            print(f'Ошибка при выполнении SQL-скрипта: {e}')

    def get_dataframe(self, query: str) -> Union[DataFrame, None]:
        """
        Выполняет SQL-запрос и возвращает результат в виде DataFrame.

        :param query: SQL-запрос (например, 'SELECT * FROM chess_users').
        :return: DataFrame с результатом запроса.
        """
        try:
            with self.engine.connect() as connection:
                # Выполнение SQL-запроса
                result = connection.execute(text(query))
                # Преобразование результата в DataFrame
                df = DataFrame(result.fetchall(), columns=result.keys())
                return df
        except exc.SQLAlchemyError as e:
            print(f'Ошибка при выполнении SQL-запроса: {e}')
            return None