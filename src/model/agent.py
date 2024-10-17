from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

from model.prompt import (
    PROMPT_CLASSIFIER,
    PROMPT_ASSISTANT,
    PROMPT_RECOMMENDER,
    PROMPT_GENERATER,
    PROMPT_SEARCH,
    PROMPT_ENTITY_MEMORY
)
from tools.utils import (
    clean_input,
    format_docs,
    format_docs_with_links,
    search_image
)


class AgentSystem:
    """AI Coocking Assistant
    """
    def __init__(self, llm, retriever, k=6):
        """Инициализация агента

        Args:
            llm: Large Language Model
            retriever: FAISS Index Retriever
            k: Количество последних сообщений в памяти агента
        """
        self.llm = llm
        self.retriever = retriever

        self.k = k

        self.chat_history = []
        self.memory_entity = self.get_memory_entity_chain()

        self.assistant_chain = self.get_assistant_chain()
        self.recommender_chain = self.get_recommender_chain()
        self.generater_chain = self.get_generater_chain()
        self.search_chain = self.get_search_chain()

        self.full_chain = self.initial_chain()

    def invoke(self, query: str):
        """Вызов цепочки

        Args:
            query (str): Запрос пользователя

        Returns:
            dict: Результат системы цепочек
        """
        result = self.full_chain.invoke({
            "input": query,
            "chat_history": self.chat_history
            })

        if result['task'] == "Unknown":
            return result

        self.chat_history.append(HumanMessage(content=query))

        if result['task'] == "Search Image":
            self.chat_history.append(AIMessage(content="Нашел изображение."))
        else:
            summary_response = self.memory_entity.invoke(result['output'])
            self.chat_history.append(AIMessage(content=summary_response))

        self.chat_history = self.chat_history[-self.k:]

        return result

    def initial_memory(self, chat_history: list):
        """Инициализация памяти для агента

        Args:
            chat_history (list): История чата
        """
        self.chat_history = chat_history

    def get_chat_history(self):
        """Возвращает текущие состояние памяти

        Returns:
            chat_history (list): История чата
        """
        return self.chat_history

    def get_memory_entity_chain(self):
        """Создание цепочки для сохранение сущностей

        Returns:
            memory_entity: Цепочка сохранения сущностей в памяти
        """
        memory_entity = (
            PromptTemplate.from_template(PROMPT_ENTITY_MEMORY)
            | self.llm
            | StrOutputParser()
        )

        return memory_entity

    def initial_chain(self):
        """Создание цепочки агентов

        Returns:
            full_chain: Итоговая цепочка системы
        """
        classifier_chain = (
            clean_input
            | PromptTemplate.from_template(PROMPT_CLASSIFIER)
            | self.llm
            | StrOutputParser()
        )

        full_chain = (
            {
                "input": itemgetter("input"),
                "topic": itemgetter("input") | classifier_chain,
                "chat_history": itemgetter("chat_history")
            }
            | RunnableLambda(self.route_chain)
        )

        return full_chain

    def get_assistant_chain(self):
        """Создание цепочки для для ассистента
        """
        assistant_chain = (
            PromptTemplate.from_template(PROMPT_ASSISTANT)
            | self.llm
            | {"output": lambda x: x.content, "task": lambda x: "About Me"}
        )

        return assistant_chain

    def get_recommender_chain(self):
        """Создание цепочки для рекомендации блюда по запросу пользователя
        """
        recommender_chain = (
            {
                "descripition": itemgetter("input")
                | self.retriever
                | format_docs_with_links,
                "query": itemgetter("input"),
                "chat_history": itemgetter("chat_history")
            }
            | PromptTemplate.from_template(PROMPT_RECOMMENDER)
            | self.llm
            | {"output": lambda x: x.content, "task": lambda x: "Recommend"}
        )

        return recommender_chain

    def get_generater_chain(self):
        """Создание цепочки для генерации идеи для нового блюда
        по запросу пользователя
        """
        generater_chain = (
            {
                "descripition": itemgetter("input")
                | self.retriever
                | format_docs,
                "query": itemgetter("input"),
                "chat_history": itemgetter("chat_history")
            }
            | PromptTemplate.from_template(PROMPT_GENERATER)
            | self.llm
            | {"output": lambda x: x.content, "task": lambda x: "Generate"}
        )

        return generater_chain

    def get_search_chain(self):
        """Создание цепочки для поиска изображения блюда
        с помощью DuckDuckGo Search
        """
        search_chain = (
            {
                "query": RunnablePassthrough(),
                "chat_history": itemgetter("chat_history")
            }
            | PromptTemplate.from_template(PROMPT_SEARCH)
            | self.llm
            | {"query": lambda x: x.content}
            | search_image
        )

        return search_chain

    def route_chain(self, dict_chain):
        """Маршрутизация запроса пользователя по цепочкам
        """
        info_class = dict_chain["topic"].lower()

        if "hello" in info_class or "about me" in info_class:
            return self.assistant_chain
        elif "recommended" in info_class:
            return self.recommender_chain
        elif "generate" in info_class:
            return self.generater_chain
        elif "image food" in info_class:
            return self.search_chain

        other_task = {
            "output": (
                "Я не понял запроса.🤯 " +
                "Попробуй переформулировать!🤔"
                ),
            "task": "Unknown"
        }

        return other_task
