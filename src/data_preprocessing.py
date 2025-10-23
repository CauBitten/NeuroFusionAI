import tensorflow as tf

def create_data_generators(
    dataset_dir,
    target_size=(299, 299),
    batch_size=20,
    validation_split=0.2,
    seed=42
):
    '''
    Cria geradores de dados para treino e validação com normalização e augmentação básica.
    
    Args:
        dataset_dir (str): Caminho da pasta raiz contendo subpastas das classes.
        target_size (tuple): Tamanho das imagens de entrada.
        batch_size (int): Tamanho do batch.
        validation_split (float): Percentual reservado para validação.
        seed (int): Semente aleatória para reprodução dos splits.

    Returns:
        train_generator, val_generator
    '''

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    # Gerador de treino
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset='training',
        shuffle=True,
        seed=seed
    )

    # Gerador de validação
    val_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset='validation',
        shuffle=False,
        seed=seed
    )

    return train_generator, val_generator
