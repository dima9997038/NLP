import uvicorn
from fastapi import FastAPI
from datasets import Dataset, ClassLabel, Value
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World!"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}!"}


@app.get("/addcsv")
async def add_csv():
    # 1. Загрузите ваш CSV файл
    df = pd.read_csv('documents.csv')  # Или создайте DataFrame из своей БД

    # 2. Преобразуйте в объект Dataset от Hugging Face
    dataset = Dataset.from_pandas(df)

    # 3. Приведем метки к единому формату (например, числам)
    # Создадим объект, который сопоставляет строковые метки с числами
    class_names = list(df['label'].unique())  # ['письмо', 'приказ']
    label_features = ClassLabel(names=class_names)

    # Функция для преобразования метки в номер
    def map_label(example):
        example['label'] = label_features.str2int(example['label'])
        return example

    dataset = dataset.map(map_label)

    # 4. Разделим данные на обучающую и проверочную выборки
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    # Теперь у нас есть dataset['train'] и dataset['test']
    return {"message": f"add csv file!"}

@app.get("/addmodel")
async def add_model():
    # Укажем название модели на Hugging Face Hub
    model_name = "ai-forever/sbert_large_nlu_ru"

    # 1. Загрузим токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Загрузим саму модель и укажем количество классов
    num_labels = 2 # 2 класса: письмо и приказ
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={0: "письмо", 1: "приказ"},  # Опционально: для красивого вывода
        label2id={"письмо": 0, "приказ": 1}
    )
    return {"message": f"add model!"}


@app.get("/tokenize")
async def say_hello(examples):
    # Укажем название модели на Hugging Face Hub
    model_name = "ai-forever/sbert_large_nlu_ru"

    # 1. Загрузим токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512)

# Применим токенизатор ко всем данным в наборе
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Куда сохранить модель
output_dir = "./my_document_classifier"

# Определим аргументы обучения
# training_args = TrainingArguments(
#     output_dir=output_dir,          # Директория для сохранения результатов
#     evaluation_strategy="epoch",    # Оценивать после каждой эпохи
#     save_strategy="epoch",          # Сохранять после каждой эпохи
#     per_device_train_batch_size=4,  # Размер батча (уменьшите, если не хватает памяти)
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,             # Количество эпох обучения
#     learning_rate=2e-5,             # Скорость обучения (очень важный параметр!)
#     weight_decay=0.01,
#     logging_dir='./logs',           # Директория для логов
#     logging_steps=10,
#     load_best_model_at_end=True,    # Загрузить лучшую модель в конце
#     metric_for_best_model="accuracy",
# )
#
# # Воспользуемся встроенной функцией для вычисления точности
# from sklearn.metrics import accuracy_score
# import numpy as np
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return {"accuracy": accuracy_score(labels, predictions)}
#
# # Создадим тренер
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
#
# # Запустим обучение!
# trainer.train()
#
# # Сохраним финальную модель
# trainer.save_model(output_dir)

# Способ 1: Использование pipeline (самый простой)
# classifier = pipeline(
#     "text-classification",
#     model=output_dir, # Используем нашу дообученную модель
#     tokenizer=tokenizer,
#     device=-1 # -1 для CPU, 0 для GPU
# )
#
# # Протестируем на новом тексте
# new_text = """
# Генеральному директору ООО «Ромашка»
# Петрову П.П.
# Исх. № 45-и от 12.05.2023
# Уважаемый Петр Петрович,
# прошу рассмотреть возможность продления договора...
# С уважением, Иванов И.И.
# """
#
# result = classifier(new_text)
# print(f"Тип документа: {result[0]['label']}, Уверенность: {result[0]['score']:.4f}")
#
# # Способ 2: Ручной способ (больше контроля)
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
#
# model = AutoModelForSequenceClassification.from_pretrained(output_dir)
# tokenizer = AutoTokenizer.from_pretrained(output_dir)
#
# inputs = tokenizer(new_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#
# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#
# predicted_class_id = predictions.argmax().item()
# confidence = predictions[0][predicted_class_id].item()
# label = model.config.id2label[predicted_class_id]
#
# print(f"Тип документа: {label}, Уверенность: {confidence:.4f}")


if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)

