import click
from .train import expressive_encoding_pipeline

@click.command()
@click.option('--config_path')
@click.option('--save_path')
def main(config_path,
        save_path):
    expressive_encoding_pipeline(config_path, save_path)

if __name__ == '__main__':
    main()
