from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from model.prompt import (
    PROMPT_CLASSIFIER,
    PROMPT_ASSISTANT,
    PROMPT_RECOMMENDER,
    PROMPT_GENERATER,
    PROMPT_SEARCH
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
    def __init__(self, llm, retriever):
        """Initital Agent

        Args:
            llm: Large Language Model
            retriever: FAISS Index Retriever
        """
        self.llm = llm
        self.retriever = retriever

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
        return self.full_chain.invoke(query)

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
            {"input": RunnablePassthrough(), "topic": classifier_chain}
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
                "query": itemgetter("input")
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
                "query": itemgetter("input")
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
            {"query": RunnablePassthrough()}
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

        if "about me" in info_class:
            return self.assistant_chain
        elif "recommended" in info_class:
            return self.recommender_chain
        elif "generate" in info_class:
            return self.generater_chain
        elif "image food" in info_class:
            return self.search_chain

        other_task = {
            "output": "К сожалению, я не понял запроса.",
            "task": "Unknown"
        }

        return other_task
