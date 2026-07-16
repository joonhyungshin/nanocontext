import click

from .evaluate import evaluate
from .gram import gram
from .markov import markov


@click.group()
def simulate():
    pass


simulate.add_command(gram)
simulate.add_command(markov)
simulate.add_command(evaluate)
