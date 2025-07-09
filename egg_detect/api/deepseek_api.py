# api/deepseek_api.py
from openai import OpenAI
from utils.logger import get_logger

class DeepSeekAPI:
    def __init__(self, api_key, base_url):
        self.logger = get_logger("DeepSeekAPI")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def analyze_egg(self, egg_data):
        try:
            prompt = (
                f"分析鸡蛋品质（80字以内）：\n"
                f"编号: {egg_data['egg_id']}\n"
                f"外壳: {egg_data['shell_status']}\n"
                f"新鲜度: {egg_data['freshness']}\n"
                f"提供品质分析及储存建议。新鲜度小于30%的不能吃"
            )
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一位鸡蛋品质分析专家，提供80字以内的品质分析及储存建议。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"DeepSeek API error: {e}")
            return f"分析失败: {str(e)}"[:30]