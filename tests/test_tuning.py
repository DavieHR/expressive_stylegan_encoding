import os
import sys
sys.path.insert(0, os.getcwd())
import pytest
import cv2

@pytest.mark.pose
def test_pose():
    from ExpressiveEncoding.train import load_model, torch, get_detector, \
                                         yaml, edict, stylegan_path, \
                                         pose_optimization, torch
    detector = get_detector()
    G = load_model(stylegan_path).synthesis
    for p in G.parameters():
        p.requires_grad = False
    pose_edit = PoseEdit()

    tuning_script_path = "./script/exp005"
    test_path = "./results/exp005/"

    tuning_result_path = "./results/exp005_tuning/pose"

    with open(os.path.join(config_path, "config.yaml")) as f:
        basis_config = edict(yaml.load(f, Loader = yaml.CLoader))

    face_folder_path = os.path.join(save_path, "data", "smooth")

    puppet_mode = False
    puppet_path = None
    if hasattr(basis_config, "puppet_path"):
        puppet_path = basis_config.puppet_path
        puppet_mode = True
    if puppet_path is not None:
        puppet_face_path = os.path.join(puppet_face_path, "smooth", "0.jpg")

    # encoder cache path
    cache_path = os.path.join(test_path, "cache.pt")
    assert os.path.exists(cache_path), "cache path not exits."

    # encoder cache
    gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(cache_path)


    # define pose loss.   
    class PoseLossRegister(LossRegisterBase):
        def forward(self, x, y):
            l2_loss = self.l2_loss(x, y).mean() * 1.0
            lpips_loss = self.lpips_loss(x,y).mean()
    
            return {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss
                   }


    face_info_from_id = get_face_info(
                                        selected_id_image,
                                        detector
                                     )

    face_info_from_gen = get_face_info(gen_image, detector)


    w_with_pose, image_posed = pose_optimization( \
                                                   selected_id_latent, \
                                                   selected_id_image, \
                                                   gen_image, \
                                                   face_info_from_gen, \
                                                   face_info_from_id, \
                                                   G, \
                                                   pose_edit, \
                                                   config_pose \
                                                 )
    image_posed = cv2.cvtColor(image_posed, cv2.COLOR_RGB2BGR)
    torch.save(w_with_pose, os.path.join(pose_path, 'latent.pt'))
    cv2.imwrite(os.path.join(pose_path, "pose.png"), image_posed * 255.0)
