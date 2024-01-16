import click
from .train import expressive_encoding_pipeline
from .puppet import puppet

@click.command()
@click.option('--pipeline', default = 'train')
@click.option('--config_path')
@click.option('--save_path')
def main(pipeline,
         config_path,
         save_path):
    if pipeline == 'train':
        expressive_encoding_pipeline(config_path, save_path)
    elif pipeline == 'puppet':
        puppet(config_path, save_path)

if __name__ == '__main__':
    main()
