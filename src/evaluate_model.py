import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import tensorflow as tf
from .data_preprocessing import create_data_generators

def evaluate_model(
    model_path='../models/inception_concat.keras',
    data_dir='../data/raw',
    output_dir='../outputs'
):
    '''
    Avalia o modelo treinado usando métricas de classificação, matriz de confusão e curvas ROC.

    Args:
        model_path (str): Caminho do arquivo de modelo (.keras).
        data_dir (str): Diretório com as imagens organizadas por classe.
        output_dir (str): Diretório para salvar plots e relatórios.

    Returns:
        dict contendo métricas principais
    '''
    # Criar pastas
    plots_dir = os.path.join(output_dir, 'plots')
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Carregar geradores
    _, val_gen = create_data_generators(data_dir, batch_size=20)

    # Carregar modelo
    model = tf.keras.models.load_model(model_path)

    # Prever probabilidades
    y_pred_proba = model.predict(val_gen, verbose=1)

    # Converter para classe predita
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes

    class_names = list(val_gen.class_indices.keys())

    # --- MATRIZ DE CONFUSÃO ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()

    # --- CLASSIFICATION REPORT ---
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # Salvar CSV
    import pandas as pd
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(reports_dir, 'classification_report.csv'))

    # --- CURVAS ROC POR CLASSE ---
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Curvas ROC por classe')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'))
    plt.close()

    # Retornar métricas principais
    return {
        'accuracy': report['accuracy'],
        'macro avg': report['macro avg'],
        'weighted avg': report['weighted avg']
    }


if __name__ == '__main__':
    metrics = evaluate_model()
    print(metrics)
