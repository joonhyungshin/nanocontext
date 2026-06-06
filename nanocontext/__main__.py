import click

from nanocontext.commands import evaluate, generate, simulate, train


@click.group()
def cli():
    pass


cli.add_command(evaluate)
cli.add_command(generate)
cli.add_command(simulate)
cli.add_command(train)
cli()
