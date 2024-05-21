"""pose score.
"""
import os
import click
import json
import cv2
import numpy as np

from .train import LossRegisterBase, torch, load_model,\
                   yaml, edict, tqdm, \
                   to_tensor, get_detector, get_face_info, \
                   stylegan_path, points, from_tensor

DEBUG = os.environ.get('DEBUG', False)
DEBUG = True if DEBUG in ['True', 'TRUE', True, 1] else False

def pose_score(
                training_path: str,
                config_path: str,
                snapshots_path: str,
                to_path: str
              ):

    latent_folder = os.path.join(snapshots_path, 'pose')
    G = load_model(stylegan_path).synthesis
    for p in G.parameters():
        p.requires_grad = False
    id_path = os.path.join(training_path, "cache.pt")
    gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(id_path)

    class PoseLossRegister(LossRegisterBase):
        def forward(self, x, y, mask):
            x = mask * x
            y = mask * y
            l2_loss = self.l2_loss(x, y).mean()
            lpips_loss = self.lpips_loss(x,y).mean()
    
            return {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss
                   }
    with open(os.path.join(config_path)) as f:
        config_pose = edict(yaml.load(f, Loader = yaml.CLoader))
    loss_register = PoseLossRegister(config_pose)
 
    if DEBUG:
        pbar = tqdm(gen_file_list[:10])
    else:
        pbar = tqdm(gen_file_list)
    detector = get_detector()
    score = {}

    for ii, _file in enumerate(pbar):

        gen_image = cv2.imread(_file)
        assert gen_image is not None, "file not exits, please check."
        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
        face_info = get_face_info(
                                   np.uint8(gen_image),
                                   detector
                                 )
        h,w = gen_image.shape[:2]
        mask_gt = np.zeros((h,w,1))
        landmarks_gt = np.int32(face_info.landmarks)
        points_gt = np.array([landmarks_gt[x[0],:] for x in points]).astype(np.int32)
        mask_gt = cv2.fillPoly(mask_gt, np.int32([points_gt]), (1,1,1))

        pad = 50
        mask_facial = np.ones((1024,1024,1), dtype = np.float32)
        pad_x = pad - 10
        pad_mouth = pad - 20
        mask_facial[310 + pad:556 - pad, 258 + pad_x: 484 - pad_x] = 0
        mask_facial[310 + pad:558 - pad, 536 + pad_x: 764 - pad_x] = 0
        mask_facial[620 + pad:908 - pad, 368 + pad_mouth: 656 - pad_mouth] = 0

        mask_gt = mask_gt * mask_facial
        mask_gt_tensor = to_tensor(mask_gt).to("cuda")

        gen_image = np.float32(gen_image / 255.0)
        gen_image = (gen_image - 0.5) * 2
        gt_tensor = to_tensor(gen_image).to("cuda")
        latent = torch.load(os.path.join(latent_folder, f"{ii + 1}.pt")).to("cuda")

        with torch.no_grad():
            fake_image = G(latent)
        loss = loss_register(fake_image, gt_tensor, mask_gt_tensor, is_gradient = False)
        
        fake_image_has_mask = fake_image * mask_gt_tensor
        fake_image_has_mask = (fake_image_has_mask + 1) * 0.5

        gen_image_has_mask = gt_tensor * mask_gt_tensor
        gen_image_has_mask = (gen_image_has_mask + 1) * 0.5

        fake_image = from_tensor(fake_image_has_mask) * 255.0
        gen_image = from_tensor(gen_image_has_mask) * 255.0

        dir_path = os.path.dirname(to_path)
        cv2.imwrite(os.path.join(dir_path, f"{ii + 1}.jpg"), np.concatenate((fake_image, gen_image), axis = 0)[...,::-1])

        score[f'{ii+ 1}'] = dict(loss = loss['loss'].item())
    with open(to_path, 'w') as f:
        json.dump(score, f)

@click.command()
@click.option('--training_path')
@click.option('--config_path')
@click.option('--snapshots_path')
@click.option('--to_path')
def _invoker(training_path,
             config_path,
             snapshots_path,
             to_path
            ):
    return pose_score(training_path, config_path, snapshots_path, to_path)

if __name__ == '__main__':
    _invoker()


