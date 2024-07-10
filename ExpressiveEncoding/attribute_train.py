"""attribute train.
"""
import os
import sys
import click

from .train import facial_attribute_optimization, torch, get_detector, get_face_info,\
                   edict, yaml, load_model, \
                   StyleSpaceDecoder, logger, stylegan_path, \
                   LossRegisterBase, np, tqdm, \
                   cv2, DEBUG

def finetune_attribute(
                        config_path: str,
                        from_path : str,
                        to_path : str
                      ):
    """finetune attribute.
    """
    class FacialLossRegister(LossRegisterBase):
        def forward(self, 
                    x, 
                    y,
                    mask,
                    weights_all,
                    weights,
                    x_pre = None,
                    y_pre = None
                   ):
        

            l2_loss = self.l2(x, y) * weights_all
            lpips_loss = self.lpips(x,y, is_reduce = False) * weights
            fp_loss = self.fp(x, y, mask)    
            inter_frame_loss = torch.zeros_like(lpips_loss)
            id_loss = self.id_loss(x, y) * 0.0
            ret = {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss,
                     "fp_loss": fp_loss,
                     "id_loss": id_loss,
                   }
            if x_pre is not None and y_pre is not None:
                inter_frame_loss = self.if_loss(
                                                x,
                                                x_pre, 
                                                y,
                                                y_pre,
                                                self.lpips
                                               ) * 2.0 * weights_all
                ret["diff_frame_loss"] = inter_frame_loss

            return ret
    # get path
    e4e_path = os.path.join(from_path, "cache.pt")
    pose_latent_path = os.path.join(from_path, "pose")

    device = "cuda:0"

    os.makedirs(to_path, exist_ok = True)
    writer = None
    if DEBUG:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard = os.path.join(to_path,  "tensorboard")
        writer = SummaryWriter(tensorboard)

    
    # get e4e files.
    gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(e4e_path)
    # loss configure.
    with open(config_path) as f:
        config_facial = edict(yaml.load(f, Loader = yaml.CLoader))
    facial_loss_register = FacialLossRegister(config_facial)


    detector = get_detector()

    face_info_from_id = get_face_info(
                                        np.uint8(selected_id_image),
                                        detector
                                     )


    gammas = None
    images_tensor_last = None
    gt_images_tensor_last = None

    p_bar = tqdm(gen_file_list)

    # init Style Space Decoder.
    from copy import deepcopy
    G = load_model(stylegan_path).synthesis
    for p in G.parameters():
        p.requires_grad = False
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(G))
    for p in ss_decoder.parameters():
        p.requires_grad = False

    latents_path = os.path.join(to_path, "latents")
    expressive_path = os.path.join(to_path, "expressive")

    for ii, _file in enumerate(p_bar):
        gen_image = cv2.imread(_file)
        assert gen_image is not None, "file not exits, please check."
        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)

        w_with_pose = torch.load(os.path.join(pose_latent_path, f"{ii + 1}.pt")).to(device)

        face_info_from_gen = get_face_info(gen_image, detector)
        style_space_latent, images_tensor_last, gt_images_tensor_last, gammas, image_gen, facial_param = \
                facial_attribute_optimization(w_with_pose, \
                                              gen_image,\
                                              face_info_from_gen,\
                                              facial_loss_register, \
                                              ss_decoder,\
                                              gammas,\
                                              images_tensor_last,\
                                              gt_images_tensor_last, \
                                              writer = writer,
                                              index = ii + 1
                                             )


        

        torch.save([x.detach() for x in style_space_latent], os.path.join(latents_path, f"{ii+1}.pt"))       
        torch.save(facial_param, os.path.join(expressive_path, f"{ii+1}.pt"))       

@click.command()
@click.option("--config_path")
@click.option("--from_path")
@click.option("--to_path")
def invoker(
            config_path : str,
            from_path : str,
            to_path : str,
           ):

    return finetune_attribute(config_path, from_path, to_path)
    
if __name__ == "__main__":
    invoker()

