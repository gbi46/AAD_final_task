"""Локальна база туристичних знань ChromaDB з детермінованими hash-embeddings.

Документи мають стабільні ідентифікатори для цитування у RAG-відповідях."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


KNOWLEDGE_DOCUMENTS = [
    "Для коротких туристичних поїздок до країн Шенгенської зони громадяни України з біометричним паспортом можуть користуватися безвізовим режимом до 90 днів протягом 180-денного періоду. Безвіз не надає права на працевлаштування.",
    "Париж найкомфортніше відвідувати у квітні–червні або вересні–жовтні. У ці періоди температура помірна, а черги до основних пам’яток зазвичай менші, ніж у липні та серпні.",
    "Метро Парижа охоплює центр міста та працює приблизно з 05:30 до 00:30. Квитки слід зберігати до завершення поїздки, а актуальну вартість перевіряти на офіційному сайті перевізника.",
    "Лувр зазвичай не працює у вівторок. Для відвідування музею варто заздалегідь обрати часовий слот і перевірити актуальний графік перед поїздкою.",
    "У Берліні діє зональна система громадського транспорту. Більшість центральних пам’яток розташована в зонах A і B; квиток потрібно активувати до початку поїздки, якщо він не цифровий.",
    "Для дощового дня у Львові підійдуть музей етнографії, аптека-музей та кавові заклади в історичному центрі. Між локаціями зручно пересуватися пішки або трамваєм.",
    "В Одесі влітку важливо планувати прогулянки на ранок або вечір, користуватися сонцезахистом і мати питну воду. Під час повітряної тривоги потрібно перейти до найближчого офіційного укриття.",
    "Перед бронюванням готелю перевірте політику скасування, фінальну суму з податками, час заселення та рейтинг об’єкта. Невідновне бронювання доцільно підтверджувати лише після перевірки дат.",
    "Базове туристичне страхування має покривати невідкладну медичну допомогу, госпіталізацію та репатріацію. Умови активного відпочинку й наявні хронічні захворювання часто потребують додаткового покриття.",
    "Для зменшення ризику кишенькових крадіжок документи й основні кошти варто тримати окремо. У людних місцях рюкзак краще переносити перед собою, а копії документів зберігати в захищеному цифровому сховищі.",
]


class LocalHashEmbedding(EmbeddingFunction[Documents]):
    """Відтворювані локальні embeddings на основі токенів і символьних триграм."""

    dimension = 192

    def __init__(self) -> None:
        self.dimension = 192

    @staticmethod
    def name() -> str:
        return "local-hash-embedding-v1"

    def get_config(self) -> dict[str, int]:
        return {"dimension": self.dimension}

    @staticmethod
    def build_from_config(config: dict[str, int]) -> "LocalHashEmbedding":
        embedding = LocalHashEmbedding()
        embedding.dimension = config.get("dimension", 192)
        return embedding

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def embed_query(self, input: Documents) -> Embeddings:
        return self(input)

    def _embed(self, text: str) -> list[float]:
        normalized = text.lower().replace("’", "'")
        tokens = re.findall(r"[\w'-]+", normalized, flags=re.UNICODE)
        vector = [0.0] * self.dimension
        weighted_features: list[tuple[str, float]] = []
        for token in tokens:
            weighted_features.append((f"word:{token}", 5.0))
            for length in range(3, min(6, len(token) + 1)):
                weighted_features.append((f"prefix:{token[:length]}", 3.0))
            for index in range(max(0, len(token) - 2)):
                weighted_features.append((f"tri:{token[index:index + 3]}", 2.0))
        for feature, weight in weighted_features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += weight
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


class KnowledgeBase:
    """Керує persistent collection та пошуком туристичних документів."""

    def __init__(self, path: str | Path = "chroma_db"):
        self.path = Path(path)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name="travel_knowledge",
            embedding_function=LocalHashEmbedding(),
            metadata={"description": "Перевірені туристичні довідкові матеріали"},
        )
        self._seed()

    def _seed(self) -> None:
        ids = [f"travel-{index:02d}" for index in range(1, len(KNOWLEDGE_DOCUMENTS) + 1)]
        self.collection.upsert(
            ids=ids,
            documents=KNOWLEDGE_DOCUMENTS,
            metadatas=[{"source": "travel-guide", "index": index} for index in range(1, 11)],
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, limit: int = 3) -> Sequence[str]:
        result = self.collection.query(query_texts=[query], n_results=min(limit, self.count))
        documents = result.get("documents") or [[]]
        return documents[0]

    def search_with_ids(self, query: str, limit: int = 3) -> list[tuple[str, str]]:
        """Те саме, що `search`, але разом з ID документа — потрібно, щоб
        відповідь researcher-агента могла посилатися на конкретне
        джерело, а не лише переказувати текст без атрибуції.
        """
        result = self.collection.query(query_texts=[query], n_results=min(limit, self.count))
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        return list(zip(ids[0], documents[0]))
