import click
from .train import expressive_encoding_pipeline
from .train_v3 import expressive_encoding_pipeline as expressive_encoding_pipeline_v3
from .puppet import puppet, puppet_video

@click.command()
@click.option('--pipeline', default = 'train')
@click.option('--path', default = None)
@click.option('--config_path')
@click.option('--save_path')
def main(pipeline,
         path,
         config_path,
         save_path):
    if pipeline == 'train':
        expressive_encoding_pipeline(config_path, save_path, path)
    if pipeline == 'train_v3':
        expressive_encoding_pipeline_v3(config_path, save_path, path)
    elif pipeline == 'puppet':
        puppet(config_path, save_path, path)
    elif pipeline == 'puppet_video':
        puppet_video(config_path, save_path, path)

if __name__ == '__main__':
    main()
