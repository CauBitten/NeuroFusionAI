import os
import scipy.io as sio
import numpy as np
import cv2
import h5py
from tqdm import tqdm

def load_mat_file(path):
    '''
    Lê um arquivo .mat, detectando se é v7.3 (HDF5) ou padrão antigo.
    Retorna dicionário com label e imagem.
    '''
    try:
        # Tenta ler como arquivo MATLAB padrão
        data = sio.loadmat(path)
        cjdata = data["cjdata"]
        label = int(cjdata["label"][0][0])
        image = cjdata["image"][0][0]
        return label, image
    
    except NotImplementedError:
        # Caso seja v7.3 (HDF5)
        with h5py.File(path, 'r') as f:
            # Os dados vêm transpostos — precisamos inverter os eixos
            label = int(f['cjdata']['label'][0][0])
            image = np.array(f['cjdata']['image']).T 
        return label, image


def convert_mat_to_images(mat_dir, output_dir):
    '''
    Converte arquivos .mat (v7 e v7.3) do dataset de Jun Cheng em imagens PNG organizadas por classe.
    '''
    os.makedirs(output_dir, exist_ok=True)
    label_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}

    for cls in label_map.values():
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]
    print(f"Encontrados {len(mat_files)} arquivos .mat")

    for file in tqdm(mat_files):
        path = os.path.join(mat_dir, file)
        try:
            label, image = load_mat_file(path)

            # Normalizar a imagem para 0–255 (uint8)
            img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            img = np.uint8(img)

            class_name = label_map[label]
            save_dir = os.path.join(output_dir, class_name)
            base_name = os.path.splitext(file)[0]
            save_path = os.path.join(save_dir, f"{base_name}.png")

            cv2.imwrite(save_path, img)

        except Exception as e:
            print(f"Erro ao processar {file}: {e}")

    print("Conversão concluída!")
