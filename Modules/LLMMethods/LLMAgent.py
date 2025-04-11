import re

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.llms import SparkLLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableParallel
import os

os.environ["AZURE_OPENAI_API_KEY"] = "e51a662bf2934ff585b9e53b21b7f6c2"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://feng-cloud-openai.openai.azure.com/"

os.environ["IFLYTEK_SPARK_APP_ID"] = "55713da8"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "NzI1OThjNDI0ODM5M2NiODBhY2NlYjFj"
os.environ["IFLYTEK_SPARK_API_KEY"] = "966252a76081be0e92e8fb0d00e7c858"


#
# os.environ["QIANFAN_AK"] = "H3cep7ar3Php6fNcNonpAubc"
# os.environ["QIANFAN_SK"] = "k5CCykaXcU80DfSdFf1iE3jAPm3iIWTY"

class LLMAgent:
    # 基础大模型Agent，继承自AgentBase
    # 使用方法：
    #   生成LLMAgent实例，调用get_response函数。为保证该类的通用性，目前未内置提示词。提示词包括系统提示词与用户提示词。系统提示词可以为空。
    #   当开启对话记忆时，需要传入参数is_first_call给出是否为第一次调用。对话记忆不记录系统提示词，系统提示词需要在每一次调用时都给出，。

    # 构造参数：
    #   agent_name*:str,agent名称，也可以使用ID替代，是区分agent对话记忆的唯一标识
    #   has_chat_history：布尔值，决定是否开启对话历史记忆，开启后agent会记住之前对其的所有询问与回答，默认开启。
    #   llm_model: str,调用大模型，目前支持“ChatGPT”，“Spark”
    #   online_track：bool,是否开启langsmith线上追踪
    #   json_format：bool,是否以json格式做出回答
    #   system_prompt = ''
    def __init__(self,
                 agent_name,
                 has_chat_history=True,
                 llm_model="ChatGPT",
                 online_track = False,
                 json_format = True,
                 system_prompt = ''
                 ):
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.has_chat_history = has_chat_history
        self.llm_model = llm_model
        self.online_track = online_track
        self.json_format = json_format
    #   调用参数
    #   system_prompt:str,系统提示词
    #   user_prompt:str,用户提示词
    #   input_param_dict:参数列表字典，该字典可以替换prompt中的待定参数
    #   is_first_call:布尔值，若为第一次调用，则清空该agent_name对应的数据库。否则继承对话记忆
    def get_response(self, user_template, new_system_template = None,input_param_dict=None, is_first_call=False):
        if input_param_dict is None:
            input_param_dict = {}
        if new_system_template is None:
            system_template = self.system_prompt
        else:
            system_template = new_system_template
        if self.online_track:
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
            os.environ['LANGCHAIN_API_KEY'] = "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"
            os.environ['LANGCHAIN_PROJECT'] = "MACE"
            # os.environ["LANGCHAIN_TRACING_V2"] = "true"
            # os.environ["LANGCHAIN_API_KEY"] = "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"
        # 1. Create prompt template

        if self.json_format:
            user_template += "\nPlease give your response in JSON format.Return a JSON object."
        if self.has_chat_history:
            system_template = PromptTemplate.from_template(system_template).invoke(input_param_dict).to_string()
            user_template = PromptTemplate.from_template(user_template).invoke(input_param_dict).to_string()
            prompt_template = ChatPromptTemplate.from_messages([
                ('system', system_template),
                MessagesPlaceholder(variable_name="history"),
                ('user',  user_template),
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ('system', system_template),
                ('user', user_template),
            ])
        # prompt_template.invoke(input_param_dict)
        # 2. Create model
        if self.llm_model == 'ChatGPT':
            model = AzureChatOpenAI(
                azure_deployment="gpt-35-turbo",
                openai_api_version="2024-02-15-preview",
                # response_format={"type": "json_object"},
            )
        if self.llm_model == 'Spark':
            model = SparkLLM(
                api_url='ws://spark-api.xf-yun.com/v1.1/chat',
                model='lite'
            )
        if self.llm_model == 'deepseek-r1:32b':
            # 初始化模型接口
            model = OllamaLLM(model="deepseek-r1:32b")
        if self.llm_model == 'qwen2.5:32b':
            # 初始化模型接口
            model = OllamaLLM(model="qwen2.5:32b")
        if self.llm_model == 'qwq:32b':
            # 初始化模型接口
            model = OllamaLLM(model="qwq:32b")
        if self.llm_model == 'deepseek-70B':
            # 初始化模型接口
            model = ChatOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model='deepseek-r1-distill-llama-70b-awq-4bit',
            )
        # 7. Create parser
        if self.json_format:
            parser = JsonOutputParser()
        else:
            parser = StrOutputParser()
        # 8. Create chain
        if self.has_chat_history:
            def get_session_history(session_id):
                return SQLChatMessageHistory(session_id, f"sqlite:///{self.agent_name}ChatMemory.db")

            if is_first_call:
                SQLChatMessageHistory(self.agent_name, f"sqlite:///{self.agent_name}ChatMemory.db").clear()
            chain = (prompt_template | model)
            runnable_with_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )|parser
            result = runnable_with_history.invoke(
                {'input': user_template},
                config={"configurable": {"session_id": self.agent_name}},
            )
            # result = parser.invoke(result)
        else:
            if self.llm_model == 'deepseek-r1:32b':
                try:
                    chain = prompt_template| model
                    result = chain.invoke(input_param_dict)
                    pattern = r"<think>(.*?)</think>"
                    think = re.findall(pattern, str(result), re.DOTALL)[0]
                    print(think)
                    result = re.sub(pattern, '', str(result))
                    result = parser.invoke(result)
                    print(result)
                except Exception as e:
                    print(e)
                    return '',think

            else:

                chain = (prompt_template|
                         model|
                         parser)
                result = chain.invoke(input_param_dict)
                print(result)
        if self.llm_model == 'deepseek-r1:32b':
            return result,think
        else:
            return result
