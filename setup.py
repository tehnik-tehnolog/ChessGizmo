# @title setup
from setuptools import setup

setup(
    name='chessgizmo',
    version='1.0',
    description='A module',
    author='tehnik-tehnolog',
    author_email='mail@mail.mail',
    packages=['chessgizmo'],  # same as name
    install_requires=['chess', 'pandas', 'stockfish', 'berserk', 'io', 'mureq',
                      'sqlalchemy', 'matplotlib', 'numpy', 'seaborn', 'plotnine',
                      'psycopg2-binary', 'aioboto3', 'python-dotenv']  # external packages as dependencies
)

