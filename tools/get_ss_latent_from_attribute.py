"""get style space latent codes script from attribute.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click
import re

from ExpressiveEncoding.train import StyleSpaceDecoder, Encoder4EditingWrapper, cv2, \
                                     e4e_path, stylegan_path, np,\
                                     to_tensor, torch, logger, \
                                     alphas

def update_alpha(dlatents, alpha_tensor):    
    dlatents_tmp = [dlatent.clone() for dlatent in dlatents]
    count = 0
    # first 5 elements.
    for k, v in alphas:
        for i in v:
            dlatents_tmp[k][:, i] = dlatents[k][:, i] + alpha_tensor[count]
            count += 1
    return dlatents_tmp

@click.command()
@click.option('--attribute_path')
@click.option('--pose_latent_path')
@click.option('--to_path')
def get_ss_latents_from_attribute(
                                  attribute_path: str,
                                  pose_latent_path: str,
                                  to_path: str
                                 ):

    os.makedirs(to_path, exist_ok = True)
    files = sorted(os.listdir(attribute_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))
    files = [os.path.join(attribute_path, x) for x in files]

    p_bar = tqdm.tqdm(files)

    decoder = StyleSpaceDecoder(stylegan_path = stylegan_path)

    latent_list = []

    os.makedirs(to_path, exist_ok = True)
    for i, _file in enumerate(p_bar):
        param = torch.load(_file)
        latent = torch.load(os.path.join(pose_latent_path, f'{i + 1}.pt'))
        ss_latent= decoder.get_style_space(latent)
        #ss_latent_updated = update_alpha(ss_latent, param[1])
        #ss_latent_updated = [x.detach().cpu() for x in ss_latent_updated]
        torch.save(ss_latent_updated, os.path.join(to_path, f'{i + 1}.pt'))
        #latent_list.append(ss_latent)
    #logger.info(f"latent saved into {to_path}.")
    #torch.save(latent_list, to_path)

if __name__ == '__main__':
    get_ss_latents_from_attribute()









