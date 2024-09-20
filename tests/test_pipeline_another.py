"""this function is used as the unit test for train module."""
# pylint: disable=no-member
# pylint: disable=import-error
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-locals
import os
import sys
import pytest
import cv2

from copy import deepcopy
sys.path.insert(0, os.getcwd())

from ExpressiveEncoding.train_another import (
                                              get_face_info,
                                              select_id_latent,
                                              pose_optimization,
                                              facial_attribute_optimization,
                                              expressive_encoding_pipeline,
                                              PoseEdit, get_detector, pipeline_init,
                                              StyleSpaceDecoder, LossRegisterBase,
                                              get_facial_gamma, pivot_finetuning
                                             )
@pytest.mark.face_info
def test_face_info():
    """ test get_face_info function.
    """
    detector = get_detector()
    image = cv2.imread("tests/face/164.png")
    assert image is not None, "image not exist."
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # pylint: disable=no-member
    face_info = get_face_info(image, detector)
    print(face_info)

@pytest.mark.init
def test_pipeline_init():
    """ test pipeline init.
    """
    save_path = "./tests/pipeline"
    config_path = './tests/'
    attachment_info = pipeline_init(config_path, save_path)
    print(attachment_info)

@pytest.mark.encoder
def test_encoder():
    """test encoder function.
    """
    from ExpressiveEncoding.train import stylegan_path,  load_model, torch # pylint: disable=import-error
    G = load_model(stylegan_path).synthesis
    save_path = "./tests/pipeline"
    config_path = './tests/'
    attachment_info = pipeline_init(config_path, save_path)
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(G))
    gen_files_list, selected_id_image, latent, selected_id = select_id_latent(
                                                                              attachment_info["w_path"],\
                                                                              attachment_info["face_folder_path"],\
                                                                              attachment_info["stage_one_path"], \
                                                                              ss_decoder,\
                                                                              "cuda:0" \
                                                                             )
    torch.save([
                gen_files_list, 
                selected_id_image,
                latent, 
                selected_id 
               ],
               attachment_info["cache_path"]
              )

@pytest.mark.pose
def test_pose():
    """test pose pipeline
    """
    from ExpressiveEncoding.train import load_model, torch, get_detector, yaml, edict, stylegan_path
    detector = get_detector()
    generate_style_gan = load_model(stylegan_path).synthesis
    for p in generate_style_gan.parameters():
        p.requires_grad = False
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(generate_style_gan))
    pose_edit = PoseEdit()

    face_e4e_path = './tests/e4e/0_gen.png'
    id_latent_path = './tests/e4e/id_latent.pt'
    id_image_path = './tests/e4e/4_gen.png'
    pose_path = "./tests/pose"
    os.makedirs(pose_path, exist_ok = True)
    selected_id_latent = torch.load(id_latent_path)
    selected_id_image = cv2.imread(id_image_path)
    gen_image = cv2.imread(face_e4e_path)
    selected_id_image = cv2.cvtColor(selected_id_image, cv2.COLOR_BGR2RGB)
    gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)

    face_info_from_id = get_face_info(
                                        selected_id_image,
                                        detector
                                     )

    face_info_from_gen = get_face_info(gen_image, detector)

    with open('./tests/pose.yaml', encoding = "utf-8") as f:
        config_pose = edict(yaml.load(f, Loader = yaml.CLoader))
    
    class PoseLossRegister(LossRegisterBase):

        def forward(self, x, y, mask):
            x = x * mask
            y = y * mask
            l2_loss = self.l2_loss(x, y).mean()
            #l2_loss = self.l2_loss(x, y).mean()
            lpips_loss = self.lpips_loss(x,y).mean()
    
            return {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss
                   }
    pose_loss_register = PoseLossRegister(config_pose)
    w_with_pose, image_posed, _ = pose_optimization( \
                                                   selected_id_latent, \
                                                   selected_id_image, \
                                                   gen_image, \
                                                   face_info_from_gen, \
                                                   face_info_from_id, \
                                                   ss_decoder, \
                                                   pose_edit, \
                                                   pose_loss_register
                                                 )
    image_posed = cv2.cvtColor(image_posed, cv2.COLOR_RGB2BGR)
    torch.save(w_with_pose, os.path.join(pose_path, 'latent.pt'))
    cv2.imwrite(os.path.join(pose_path, "pose.png"), image_posed * 255.0)

@pytest.mark.gamma
def test_gamma():
    """get facial gamma
    """
    import numpy as np
    from copy import deepcopy
    from ExpressiveEncoding.train_another import load_model, torch, get_detector, \
                                         yaml, edict, stylegan_path, \
                                         from_tensor, StyleSpaceDecoder, \
                                         LossRegisterBase, Timer, \
                                         gen_masks, region_names
    from ExpressiveEncoding.loss import LPIPS
    device = "cuda:0"
    generate_style_gan = load_model(stylegan_path).synthesis
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(generate_style_gan))
    ss_decoder.to(device)

    id_image_path = './tests/e4e/4_gen.png'
    selected_id_image = cv2.imread(id_image_path)

    id_latent_path = './tests/e4e/id_latent.pt'
    selected_id_latent = torch.load(id_latent_path).to(device)
    dlatents = ss_decoder.get_style_space(selected_id_latent)
    detector = get_detector()
    face_info_from_id = get_face_info(
                                      np.uint8(selected_id_image),
                                      detector
                                     )

    masks = gen_masks(face_info_from_id.landmarks, selected_id_image)
    gammas = get_facial_gamma(
                              dlatents,
                              masks,
                              ss_decoder,
                              LPIPS(),
                              device
                             )


@pytest.mark.facial_attribute
def test_facial_attribute():
    """facial attribute optimization pipeline
    """
    from copy import deepcopy
    from ExpressiveEncoding.train_another import load_model, torch, get_detector, \
                                          yaml, edict, stylegan_path, \
                                          from_tensor, StyleSpaceDecoder, \
                                          LossRegisterBase, Timer, \
                                          gen_masks, region_names, \
                                          get_face_info
    from ExpressiveEncoding.loss import LPIPS

    import numpy as np
    t = Timer()
    device = "cuda:0"
    detector = get_detector()
    generate_style_gan = load_model(stylegan_path).synthesis
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(generate_style_gan)).to(device)
    face_e4e_path = './tests/e4e/0_gen.png'
    id_latent_path = './tests/pose/latent.pt'
    pose_path = "./tests/facial_attribute"
    os.makedirs(pose_path, exist_ok = True)
    id_latent = torch.load(id_latent_path)
    id_image_path = './tests/e4e/4_gen.png'
    selected_id_image = cv2.imread(id_image_path)
    gen_image = cv2.imread(face_e4e_path)
    gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
    face_info_from_id = get_face_info(
                                      np.uint8(selected_id_image),
                                      detector
                                     )
    masks = gen_masks(face_info_from_id.landmarks, selected_id_image)
    dlatents = ss_decoder.get_style_space(id_latent)

    gammas = get_facial_gamma(
                              dlatents,
                              masks,
                              ss_decoder,
                              LPIPS(),
                              device
                             )
    with open('./tests/facial_attribute.yaml', encoding = 'utf-8') as f:
        config_facial = edict(yaml.load(f, Loader = yaml.CLoader))
    face_info_from_gen = get_face_info(
                                        gen_image,
                                        detector
                                      )
    class FacialLossRegister(LossRegisterBase):

        def forward(
                    self,
                    x,
                    y,
                   ):
            n = x[0].shape[0]
            l1_loss = torch.zeros((n)).to(x[0])
            for (_x, _y) in zip(x,y):
                l1_loss += self.l1_loss(_x, _y)
            ret = {
                    "l1_loss": l1_loss
                  }

            return ret

    loss_register = FacialLossRegister(config_facial)
    _, image_gen, _ = facial_attribute_optimization( \
                                                     id_latent, \
                                                     gen_image, \
                                                     face_info_from_gen, \
                                                     loss_register, \
                                                     ss_decoder, \
                                                     gammas = gammas
                                                    )
    cv2.imwrite("./tests/facial.png", image_gen[...,::-1])

@pytest.mark.pti
def test_pti():
    """pti training pipeline unit test.
    """
    from copy import deepcopy
    from ExpressiveEncoding.train import load_model, yaml, edict, stylegan_path, StyleSpaceDecoder
    from tensorboardX import SummaryWriter
    generate_style_gan = load_model(stylegan_path).synthesis
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(generate_style_gan))

    face_folder = './results/exp010/0/data/smooth'
    style_latent_path = './results/exp010/0/facial_cpu'
    config_path = './tests/pti.yaml'

    snapshots = './tests/pti'
    os.makedirs(snapshots, exist_ok = True)

    with open(config_path, encoding = 'utf-8') as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    writer = SummaryWriter("./tests/pti/tensorboard/")

    _ = pivot_finetuning(
                         face_folder, \
                         style_latent_path, \
                         snapshots,\
                         ss_decoder, \
                         config, \
                         writer = writer
                        )

@pytest.mark.validate
def test_validate():
    """the validate function 
    """
    from copy import deepcopy
    from ExpressiveEncoding.train import load_model, stylegan_path, \
                                  StyleSpaceDecoder, validate_video_gen
    generate_style_gan = load_model(stylegan_path).synthesis
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(generate_style_gan))
    state_dict_path = './results/exp002/pti/snapshots/200.pth'
    latent_folder = './results/exp002/facial'
    save_path = "./tests/validate.mp4"
    validate_video_gen(
                       save_path,
                       state_dict_path,
                       latent_folder,
                       ss_decoder,
                       100
                      )

@pytest.mark.pipeline
def test_pipeline():
    """the pipeline of Expressive Encoding codes.
    """
    save_path = "./tests/pipeline"
    config_path = './tests/'
    expressive_encoding_pipeline(config_path, save_path)

