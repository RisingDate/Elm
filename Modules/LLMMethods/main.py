import ollama
from LLMAgent import LLMAgent
import pandas as pd
path = '../../Dataset/train.txt'
data = pd.read_csv(path, sep="\t")
print(data.columns)
df = data[['title', 'content', 'gender', 'age', 'city', 'fans_cnt', 'topics', 'video_cnt', 'coin_cnt',
           'post_type', 'cover_ocr_content', 'video_content']]
system_prompt = '''
你是一个营销洞察与舆情监控领域的专家，你需要利用你的知识对推文各项信息进行分析解读，最后根据这些信息给出你对其的预测互动量（转发、评论、点赞）。
title表示素材标题，content表示素材内容，配套图片/视频的说明，gender表示坐着性别，age作者年龄，city作者城市,fans_cnt作者粉丝数
topics主题列表,video_cnt作者视频数,coin_cnt作者所获点赞数, post_type主帖类型, cover_ocr_content视频封面内容识别,video_content视频文本识别
'''
agent = LLMAgent(agent_name='X',has_chat_history=False,llm_model='deepseek-r1:32b',system_prompt=system_prompt)
user_data = df.iloc[0]
user_prompt = f'''
以下为你需要预测的推文数据，请给出你的预测互动量.最终结果应为一个整数。
{user_data}
'''
ans = agent.get_response(user_prompt)
print(df.iloc[0])
