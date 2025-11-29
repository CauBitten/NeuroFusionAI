import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from .data_preprocessing import create_data_generators
from .model_inception_concat import build_inception_concat_model

def train_model(data_dir='../data/raw',
                batch_size=20,
                epochs=50,
                model_dir='../models',
                log_dir='../outputs/logs'):
    '''
    Treina o modelo Inception-v3 com concatenação das features.
    '''

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)

    # Criar geradores
    train_gen, val_gen = create_data_generators(data_dir, batch_size=batch_size)

    # Criar modelo
    model = build_inception_concat_model(num_classes=len(train_gen.class_indices))

    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint_path = os.path.join(model_dir, 'checkpoints', 'inception_concat_best.h5')
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        CSVLogger(os.path.join(log_dir, 'training_log.csv'), append=True)
    ]

    # Treinar modelo
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # Salvar modelo final e histórico
    model.save(os.path.join(model_dir, 'inception_concat.keras'))

    return history, model


if __name__ == '__main__':
    train_model()
