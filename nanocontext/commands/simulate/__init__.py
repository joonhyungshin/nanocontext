import click

from .gram import gram
from .markov import markov


@click.group()
def simulate():
    pass


simulate.add_command(gram)
simulate.add_command(markov)
