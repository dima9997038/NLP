# -*- coding: utf-8 -*-
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
# Если у вас нет готового CSV, создаем демо-датасет
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


# Создаем или загружаем данные
try:
    df = pd.read_csv('documents.csv')
    print("Загружен documents.csv")
except FileNotFoundError:
    print("Файл documents.csv не найден, создаем демо-данные")
    df = create_demo_data()
    df.to_csv('documents.csv', index=False, encoding='utf-8')

print(f"Размер датасета: {len(df)} документов")
print(f"Распределение классов:\n{df['label'].value_counts()}")

# ==================== 2. ПОДГОТОВКА ДАННЫХ ====================
# Создаем Dataset объект
dataset = Dataset.from_pandas(df)

# Определяем классы
class_names = list(df['label'].unique())
print(f"Классы: {class_names}")

label_features = ClassLabel(names=class_names)


# Преобразуем текстовые метки в числовые
def map_label(example):
    example['label'] = label_features.str2int(example['label'])
    return example


dataset = dataset.map(map_label)

# Разделяем на train/test
dataset = dataset.train_test_split(test_size=0.3, seed=42)
print(f"Обучающая выборка: {len(dataset['train'])} документов")
print(f"Тестовая выборка: {len(dataset['test'])} документов")

# ==================== 3. ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА ====================
model_name = "ai-forever/sbert_large_nlu_ru"
num_labels = len(class_names)

print(f"\nЗагружаем модель: {model_name}")
print(f"Количество классов: {num_labels}")

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загружаем модель
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={i: name for i, name in enumerate(class_names)},
    label2id={name: i for i, name in enumerate(class_names)}
)


# ==================== 4. ТОКЕНИЗАЦИЯ ДАННЫХ ====================
def tokenize_function(examples):
    """Функция для токенизации текста"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )


print("Токенизируем данные...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ==================== 5. НАСТРОЙКА ОБУЧЕНИЯ ====================
# Аргументы обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch",
)


# Функция для вычисления метрик
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# Создаем тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ==================== 6. ОБУЧЕНИЕ МОДЕЛИ ====================
print("\nНачинаем обучение модели...")
train_result = trainer.train()

# Сохраняем модель
trainer.save_model("./my_document_classifier")
tokenizer.save_pretrained("./my_document_classifier")
print("Модель сохранена в папку './my_document_classifier'")

# ==================== 7. ТЕСТИРОВАНИЕ МОДЕЛИ ====================
print("\nТестируем модель на тестовых данных...")
eval_results = trainer.evaluate()
print(f"Точность на тестовых данных: {eval_results['eval_accuracy']:.3f}")


# ==================== 8. ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================
def predict_document_type(text):
    """Функция для предсказания типа документа"""
    # Токенизируем текст
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Получаем результаты
    predicted_class_id = predictions.argmax().item()
    confidence = predictions[0][predicted_class_id].item()
    label = model.config.id2label[predicted_class_id]

    return label, confidence


# Тестовые примеры
test_texts = [
    "Уважаемый коллега, прошу вас предоставить финансовый отчет за последний месяц для подготовки к совещанию.",
    "ПРИКАЗ № 567 от 01.12.2023. О внесении изменений в штатное расписание предприятия. Приказываю утвердить новое штатное расписание."
]

print("\nТестируем на новых примерах:")
for i, text in enumerate(test_texts, 1):
    label, confidence = predict_document_type(text)
    print(f"\nПример {i}:")
    print(f"Текст: {text[:100]}...")
    print(f"Предсказание: {label}")
    print(f"Уверенность: {confidence:.3f}")


# ==================== 9. СОХРАНЕНИЕ ДЛЯ ДАЛЬНЕЙШЕГО ИСПОЛЬЗОВАНИЯ ====================
def save_classification_pipeline():
    """Сохранение pipeline для простого использования"""
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="./my_document_classifier",
        tokenizer="./my_document_classifier",
        device=-1  # -1 для CPU, 0 для GPU
    )

    # Тестируем pipeline
    result = classifier(test_texts[0])
    print(f"\nPipeline test: {result[0]}")

    return classifier


print("\nСоздаем и тестируем pipeline...")
classifier = save_classification_pipeline()

print("\n✅ Проект успешно завершен! Модель готова к использованию.")