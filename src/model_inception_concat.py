from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_inception_concat_model(input_shape=(299, 299, 3), num_classes=3):
    '''
    Cria o modelo Inception-v3 com concatenação das saídas dos blocos mixed3, mixed7 e mixed10.
    '''
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # Extrai saídas intermediárias dos blocos indicados
    layer_mixed3 = base_model.get_layer('mixed3').output
    layer_mixed7 = base_model.get_layer('mixed7').output
    layer_mixed10 = base_model.get_layer('mixed10').output

    # Global average pooling em cada saída
    gap3 = GlobalAveragePooling2D()(layer_mixed3)
    gap7 = GlobalAveragePooling2D()(layer_mixed7)
    gap10 = GlobalAveragePooling2D()(layer_mixed10)

    # Concatenação das três representações
    concatenated = Concatenate()([gap3, gap7, gap10])

    # Camadas densas finais
    x = Dropout(0.4)(concatenated)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Congelar os pesos da base inicialmente
    for layer in base_model.layers:
        layer.trainable = False

    return model
