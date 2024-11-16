import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
from transformers import (
    AlbertTokenizer,
    AlbertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AlbertConfig
)
from sklearn.metrics import classification_report
from tabulate import tabulate

# Step 0: Specify the GPU to use
GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Check and print the selected GPU
print(f"Using GPU: {GPU_ID}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Using CPU instead.")

# Create a directory to save results
RESULTS_DIR = "results_albert3"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Step 1: Load and Preprocess Data
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

# Step 2: Initialize the Tokenizer and Dataset
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
max_len = 128
dataset = SentimentDataset(
    texts=df['Sentence'].to_numpy(),
    labels=df['label'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
)

# Step 3: Split the Dataset into Training, Validation, and Test Sets
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

# Function to start training
def start_training(learning_rate, num_train_epochs, batch_size, weight_decay, patience=5):
    # Set up model configuration with weight decay (note: weight decay is handled in optimizer, not model config)
    config = AlbertConfig.from_pretrained(
        'albert-base-v2',
        num_labels=3
    )
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', config=config)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./best_model_ALBERT',
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,  # Added weight decay
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=True
    )

    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    # Train the model and capture training/validation loss
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

    # Determine fit status
    if validation_loss > training_loss and (validation_loss - training_loss) > 0.1:
        fit_status = "Overfitting"
    elif validation_loss > training_loss:
        fit_status = "Good Fit"
    else:
        fit_status = "Underfitting"

    return test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status

# Default hyperparameters
default_params = {
    'Batch Size': 32,
    'Learning Rate': 1.5e-5,
    'Epochs': 20,
    'Weight Decay': 0.0  # Default weight decay
}

# Parameters to test (including weight decay)
batch_sizes = [16, 32]  # Varying batch size
learning_rates = [1.5e-7, 1.5e-6, 5e-6, 1.5e-5, 5e-5, 1.5e-4, 5e-4, 1.5e-3, 2e-3, 5e-3, 0.01, 1.5e-2, 1.75e-2, 2e-2]  # Varying learning rate
epochs = [5, 10, 20, 50, 100, 200, 300, 400, 450, 500]  # Varying number of epochs
weight_decays = [0.0, 0.01, 0.05, 0.1, 0.2]  # Varying weight decay

def run_experiments():
    results = []

    # Test Batch Size (with other parameters fixed)
    for batch_size in batch_sizes:
        print(f"Testing Batch Size: {batch_size}")
        test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status = start_training(
            learning_rate=default_params['Learning Rate'],
            num_train_epochs=default_params['Epochs'],
            batch_size=batch_size,
            weight_decay=default_params['Weight Decay']
        )
        results.append({
            'Hyperparameter': 'Batch Size',
            'Value': batch_size,
            'Batch Size': batch_size,
            'Learning Rate': default_params['Learning Rate'],
            'Epochs': default_params['Epochs'],
            'Weight Decay': default_params['Weight Decay'],
            'Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Fit Status': fit_status
        })

    # Test Learning Rate (with other parameters fixed)
    for learning_rate in learning_rates:
        print(f"Testing Learning Rate: {learning_rate}")
        test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status = start_training(
            learning_rate=learning_rate,
            num_train_epochs=default_params['Epochs'],
            batch_size=default_params['Batch Size'],
            weight_decay=default_params['Weight Decay']
        )
        results.append({
            'Hyperparameter': 'Learning Rate',
            'Value': learning_rate,
            'Batch Size': default_params['Batch Size'],
            'Learning Rate': learning_rate,
            'Epochs': default_params['Epochs'],
            'Weight Decay': default_params['Weight Decay'],
            'Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Fit Status': fit_status
        })

    # Test Epochs (with other parameters fixed)
    for num_epochs in epochs:
        print(f"Testing Epochs: {num_epochs}")
        test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status = start_training(
            learning_rate=default_params['Learning Rate'],
            num_train_epochs=num_epochs,
            batch_size=default_params['Batch Size'],
            weight_decay=default_params['Weight Decay']
        )
        results.append({
            'Hyperparameter': 'Epochs',
            'Value': num_epochs,
            'Batch Size': default_params['Batch Size'],
            'Learning Rate': default_params['Learning Rate'],
            'Epochs': num_epochs,
            'Weight Decay': default_params['Weight Decay'],
            'Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Fit Status': fit_status
        })

    # Test Weight Decay (with other parameters fixed)
    for weight_decay in weight_decays:
        print(f"Testing Weight Decay: {weight_decay}")
        test_accuracy, precision, recall, f1_score, training_loss, validation_loss, fit_status = start_training(
            learning_rate=default_params['Learning Rate'],
            num_train_epochs=default_params['Epochs'],
            batch_size=default_params['Batch Size'],
            weight_decay=weight_decay
        )
        results.append({
            'Hyperparameter': 'Weight Decay',
            'Value': weight_decay,
            'Batch Size': default_params['Batch Size'],
            'Learning Rate': default_params['Learning Rate'],
            'Epochs': default_params['Epochs'],
            'Weight Decay': weight_decay,
            'Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Fit Status': fit_status
        })

    # Convert results to a DataFrame with a specific column order
    df_results = pd.DataFrame(results, columns=[
        'Hyperparameter', 'Value', 'Batch Size', 'Learning Rate', 'Epochs', 'Weight Decay',
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Loss',
        'Validation Loss', 'Fit Status'
    ])

    # Save results as CSV
    results_file = os.path.join(RESULTS_DIR, 'experiment_results.csv')
    df_results.to_csv(results_file, index=False)
    print(f"Saved experiment results to {results_file}")

    # Save results as a table
    table_file = os.path.join(RESULTS_DIR, 'experiment_results.txt')
    with open(table_file, 'w') as f:
        f.write(tabulate(df_results, headers='keys', tablefmt='fancy_grid', showindex=False))
    print(f"Saved experiment table to {table_file}")

    return df_results

def plot_results(df_results):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    hyperparameters = ['Batch Size', 'Learning Rate', 'Epochs', 'Weight Decay']

    # Iterate through each hyperparameter and metric to create separate plots
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

            # Format x-axis if hyperparameter is Learning Rate or Weight Decay for better readability
            if hyperparameter in ['Learning Rate', 'Weight Decay']:
                plt.xscale('log') if hyperparameter == 'Learning Rate' else plt.xscale('linear')

            # Enhance plot aesthetics
            plt.grid(True, which="both", ls="--", linewidth=0.5)

            plot_file = os.path.join(RESULTS_DIR, f'{metric.lower()}_vs_{hyperparameter.lower().replace(" ", "_")}_plot.png')
            plt.savefig(plot_file)
            print(f"Saved {metric} vs {hyperparameter} plot to {plot_file}")
            plt.close()

if __name__ == "__main__":
    experiment_df = run_experiments()

    print("\nResults After Training:")
    print(tabulate(experiment_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    plot_results(experiment_df)