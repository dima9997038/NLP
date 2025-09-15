# -*- coding: utf-8 -*-
import os
import warnings

# Отключаем предупреждение о symlinks
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')

import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score
import torch


# ==================== 1. СОЗДАНИЕ ДАТАСЕТА ====================
def create_demo_data():
    """Создание демонстрационного набора данных"""
    data = {
        'text': [
            # Письма
            "Уважаемый Иван Иванович, прошу предоставить отчет о проделанной работе за первый квартал 2023 года до 15 апреля включительно. С уважением, Петров П.П.",
            "Генеральному директору ООО «Ромашка» Петрову П.П. Исх. № 45-и от 12.05.2023. Уважаемый Петр Петрович, прошу рассмотреть возможность продления договора аренды.",
            "В отдел бухгалтерии ООО «СтройИнвест». Исх. № 189-р от 18.07.2023. Прошу произвести оплату по договору № 456 от 10.07.2023.",

            # Приказы
            "ПРИКАЗ № 145-к от 12.05.2023. О назначении ответственного за пожарную безопасность. 1. Назначить инженера по охране труда Сидорова С.С. ответственным.",
            "ПРИКАЗ № 78 от 03.04.2023. О проведении ежегодного медицинского осмотра сотрудников. В соответствии с требованиями трудового законодательства приказываю:",
            "ПРИКАЗ № 201 от 25.06.2023. Об утверждении графика отпусков на второе полугодие 2023 года. На основании статей 123 Трудового кодекса РФ приказываю:"
        ],
        'label': ['письмо', 'письмо', 'письмо', 'приказ', 'приказ', 'приказ']
    }
    return pd.DataFrame(data)


print("Создаем демо-данные...")
df = create_demo_data()

print(f"Размер датасета: {len(df)} документов")
print(f"Распределение классов:\n{df['label'].value_counts()}")

# ==================== 2. ПОДГОТОВКА ДАННЫХ ====================
dataset = Dataset.from_pandas(df)
class_names = list(df['label'].unique())
print(f"Классы: {class_names}")

label_features = ClassLabel(names=class_names)


def map_label(example):
    example['label'] = label_features.str2int(example['label'])
    return example


dataset = dataset.map(map_label)
dataset = dataset.train_test_split(test_size=0.3, seed=42)

# ==================== 3. ЗАГРУЗКА МОДЕЛИ ====================
model_name = "ai-forever/sbert_large_nlu_ru"
num_labels = len(class_names)

print(f"\nЗагружаем модель: {model_name}")
print("Это может занять несколько минут...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={0: "письмо", 1: "приказ"},
    label2id={"письмо": 0, "приказ": 1}
)


# ==================== 4. ТОКЕНИЗАЦИЯ ====================
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=256,  # Уменьшили для скорости
        return_tensors="pt"
    )


print("Токенизируем данные...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ==================== 5. ОБУЧЕНИЕ ====================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,  # Уменьшили для скорости
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    save_strategy="no",  # Отключаем сохранение для демо
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nНачинаем обучение...")
trainer.train()


# ==================== 6. ТЕСТИРОВАНИЕ ====================
def predict_document_type(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class_id = predictions.argmax().item()
    confidence = predictions[0][predicted_class_id].item()
    label = model.config.id2label[predicted_class_id]

    return label, confidence


# Тестовые примеры
test_texts = [
    "Уважаемый коллега, прошу вас предоставить финансовый отчет",
    "ПРИКАЗ № 567 от 01.12.2023. О внесении изменений в штатное расписание"
]

print("\nТестируем модель:")
for i, text in enumerate(test_texts, 1):
    label, confidence = predict_document_type(text)
    print(f"Пример {i}: '{text[:30]}...' → {label} ({confidence:.2f})")

print("\n✅ Модель успешно обучена и протестирована!")