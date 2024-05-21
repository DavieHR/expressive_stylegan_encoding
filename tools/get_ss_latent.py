"""get style space latent codes script.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click

from ExpressiveEncoding.train import StyleSpaceDecoder, Encoder4EditingWrapper, cv2, \
                                     e4e_path, stylegan_path, np,\
                                     to_tensor, torch, logger

@click.command()
@click.option('--from_path')
@click.option('--to_path')
def get_ss_latents(
                    from_path: str,
                    to_path: str
                  ):

    files = sorted(os.listdir(from_path), key = lambda x: int(x.split('.')[0]))
    files = [os.path.join(from_path, x) for x in files]

    p_bar = tqdm.tqdm(files)

    e4e = Encoder4EditingWrapper(e4e_path)
    decoder = StyleSpaceDecoder(stylegan_path = stylegan_path)

    latent_list = []


    os.makedirs(to_path, exist_ok = True)
    for i, _file in enumerate(p_bar):
        image = np.float32(cv2.imread(_file) / 255.0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256), interpolation = cv2.INTER_CUBIC)
        image_tensor = 2 * (to_tensor(image).to("cuda") - 0.5)
        with torch.no_grad():
            latent = e4e(image_tensor)
            ss_latent = decoder.get_style_space(latent)
        torch.save(ss_latent, os.path.join(to_path, f'{i + 1}.pt'))
        #latent_list.append(ss_latent)
    #logger.info(f"latent saved into {to_path}.")
    #torch.save(latent_list, to_path)

if __name__ == '__main__':
    get_ss_latents()








