import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report
from tabulate import tabulate
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np

GPU_IDS = '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

print(f"Using GPUs: {GPU_IDS}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU instead.")

RESULTS_DIR = "resulta"  
os.makedirs(RESULTS_DIR, exist_ok=True)


df = pd.read_csv("data.csv")
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df['label'] = df['Sentiment'].map(label_map)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_len = 128
dataset = SentimentDataset(
    texts=df['Sentence'].to_numpy(),
    labels=df['label'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
)

train_size = 0.7
val_size = 0.15
test_size = 0.15

train_len = int(train_size * len(dataset))
val_len = int(val_size * len(dataset))
test_len = len(dataset) - train_len - val_len

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_len, val_len, test_len]
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

def visualize_shap(shap_explainer, texts, save_dir):
    """
    Generate and save SHAP text plots for a list of texts.
    """
    shap_values = shap_explainer(texts)
    for i, text in enumerate(texts):
        if hasattr(shap_values[i], 'data'):
            shap_values[i].data = clean_tokens(shap_values[i].data)
        elif hasattr(shap_values[i], 'text'):
            shap_values[i].text = clean_tokens(shap_values[i].text)

        shap_plot_html = shap.plots.text(shap_values[i], display=False)

        shap_plot_path = os.path.join(save_dir, f'shap_explanation_{i+1}.html')
        with open(shap_plot_path, 'w', encoding='utf-8') as f:
            f.write(shap_plot_html)
        print(f"Saved SHAP explanation for sample {i+1} to {shap_plot_path}")

def clean_tokens(tokens):
    """
    Clean the tokens by removing the special 'Ġ' symbol and adjusting spaces.
    """
    cleaned_tokens = []
    for token in tokens:
        if token.startswith('Ġ'):
            cleaned_tokens.append(' ' + token[1:]) 
        else:
            cleaned_tokens.append(token)
    return cleaned_tokens

def visualize_lime(lime_explainer, texts, tokenizer, model, save_dir):
    """
    Generate and save LIME explanations for a list of texts.
    """
    for i, text in enumerate(texts):
        exp = lime_explainer.explain_instance(
            text, 
            lambda x: predict_proba_lime(x, tokenizer, model),
            num_features=10, 
            top_labels=1
        )
        lime_html_path = os.path.join(save_dir, f'lime_explanation_{i+1}.html')
        exp.save_to_file(lime_html_path)
        print(f"Saved LIME explanation for sample {i+1} to {lime_html_path}")

def predict_proba_shap(texts, tokenizer, model):
    """
    Predict probabilities for a list of texts using the tokenizer and model.
    """
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    else:
        texts = list(texts)

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

def predict_proba_lime(texts, tokenizer, model):
    """
    Predict probabilities for a list of texts using the tokenizer and model.
    """
    return predict_proba_shap(texts, tokenizer, model)

def start_training(learning_rate, num_train_epochs, batch_size, patience=5, model_save_path=None):
    training_args = TrainingArguments(
        output_dir='./best_model_RoBERTa',
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,  
        per_device_eval_batch_size=batch_size,   
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=False,         
        fp16=True,                 
        dataloader_num_workers=4,    
        run_name=f"RoBERTa_lr{learning_rate}_bs{batch_size}_epochs{num_train_epochs}"
    )

    config = RobertaConfig.from_pretrained('roberta-base', num_labels=3)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    train_result = trainer.train()
    training_loss = train_result.training_loss
    eval_result = trainer.evaluate(eval_dataset=val_dataset)
    validation_loss = eval_result['eval_loss']

    preds_output = trainer.predict(test_dataset)
    preds = preds_output.predictions.argmax(-1)
    labels = preds_output.label_ids

    classification_rep = classification_report(labels, preds, output_dict=True, target_names=["Negative", "Neutral", "Positive"])
    test_accuracy = classification_rep['accuracy']
    precision = classification_rep['weighted avg']['precision']
    recall = classification_rep['weighted avg']['recall']
    f1_score = classification_rep['weighted avg']['f1-score']

    if validation_loss > training_loss and (validation_loss - training_loss) > 0.1:
        fit_status = "Overfitting"
    elif validation_loss > training_loss:
        fit_status = "Good Fit"
    else:
        fit_status = "Underfitting"

    if model_save_path:
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)

    sample_texts_shap = df['Sentence'].sample(5, random_state=42).tolist()
    background_texts = df['Sentence'].sample(100, random_state=42).tolist()

    masker = shap.maskers.Text(tokenizer=tokenizer)

    shap_explainer = shap.Explainer(
        lambda x: predict_proba_shap(x, tokenizer, model),
        masker=masker,
        data=background_texts
    )
    visualize_shap(shap_explainer, sample_texts_shap, RESULTS_DIR)

    sample_texts_lime = df['Sentence'].sample(5, random_state=42).tolist()
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
    visualize_lime(lime_explainer, sample_texts_lime, tokenizer, model, RESULTS_DIR)

    return test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status

default_params = {
    'Batch Size': 16, 
    'Learning Rate': 1.5e-5,
    'Epochs': 20
}

batch_sizes = [16, 32]
learning_rates = [1.5e-7, 1.5e-6, 5e-6, 1.5e-5, 5e-5, 1.5e-4, 5e-4, 1.5e-3, 2e-3, 5e-3, 0.01, 1.5e-2, 1.75e-2, 2e-2] 
epochs = [5, 10, 20, 50, 100, 200, 300, 400, 450, 500] 
def run_experiments():
    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting Batch Size: {batch_size}")
        test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status = start_training(
            learning_rate=default_params['Learning Rate'], 
            num_train_epochs=default_params['Epochs'], 
            batch_size=batch_size, 
            model_save_path=os.path.join(RESULTS_DIR, f'roberta_batch_{batch_size}')
        )
        results.append({
            'Hyperparameter': 'Batch Size',
            'Value': batch_size,
            'Batch Size': batch_size,
            'Learning Rate': default_params['Learning Rate'],
            'Epochs': default_params['Epochs'],
            'Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Fit Status': fit_status
        })

    for learning_rate in learning_rates:
        print(f"\nTesting Learning Rate: {learning_rate}")
        test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status = start_training(
            learning_rate=learning_rate, 
            num_train_epochs=default_params['Epochs'], 
            batch_size=default_params['Batch Size'], 
            model_save_path=os.path.join(RESULTS_DIR, f'roberta_lr_{learning_rate}')
        )
        results.append({
            'Hyperparameter': 'Learning Rate',
            'Value': learning_rate,
            'Batch Size': default_params['Batch Size'],
            'Learning Rate': learning_rate,
            'Epochs': default_params['Epochs'],
            'Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Fit Status': fit_status
        })

    for num_epochs in epochs:
        print(f"\nTesting Epochs: {num_epochs}")
        test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status = start_training(
            learning_rate=default_params['Learning Rate'], 
            num_train_epochs=num_epochs, 
            batch_size=default_params['Batch Size'], 
            model_save_path=os.path.join(RESULTS_DIR, f'roberta_epochs_{num_epochs}')
        )
        results.append({
            'Hyperparameter': 'Epochs',
            'Value': num_epochs,
            'Batch Size': default_params['Batch Size'],
            'Learning Rate': default_params['Learning Rate'],
            'Epochs': num_epochs,
            'Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Fit Status': fit_status
        })

    df_results = pd.DataFrame(results, columns=[
        'Hyperparameter', 'Value', 'Batch Size', 'Learning Rate', 'Epochs',
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Loss',
        'Validation Loss', 'Fit Status'
    ])

    results_file = os.path.join(RESULTS_DIR, 'experiment_results.csv')
    df_results.to_csv(results_file, index=False)
    print(f"\nSaved experiment results to {results_file}")

    table_file = os.path.join(RESULTS_DIR, 'experiment_results.txt')
    with open(table_file, 'w') as f:
        f.write(tabulate(df_results, headers='keys', tablefmt='fancy_grid', showindex=False))
    print(f"Saved experiment table to {table_file}")

    return df_results

def plot_results(df_results):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    hyperparameters = ['Batch Size', 'Learning Rate', 'Epochs']

    for hyperparameter in hyperparameters:
        df_filtered = df_results[df_results['Hyperparameter'] == hyperparameter]

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_filtered, 
                x='Value', 
                y=metric, 
                marker='o'
            )
            plt.title(f'{metric} vs {hyperparameter}')
            plt.xlabel(hyperparameter)
            plt.ylabel(metric)
            plt.tight_layout()

            hyperparameter_clean = hyperparameter.lower().replace(" ", "_")
            metric_clean = metric.lower().replace(" ", "_")
            plot_file = os.path.join(RESULTS_DIR, f'{metric_clean}_vs_{hyperparameter_clean}_plot.png')
            plt.savefig(plot_file)
            print(f"Saved {metric} vs {hyperparameter} plot to {plot_file}")
            plt.close()

if __name__ == "__main__":
    experiment_df = run_experiments()

    print("\nResults After Training:")
    print(tabulate(experiment_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    plot_results(experiment_df)
