"""from openai import OpenAI

client=OpenAI(api_key="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")

try:
    response = client.models.list()
    print(response)
    print("✅ API key works. Models:", [model.id for model in response.data])
except Exception as e:
    print("❌ API Key issue:", e)"""

from openai import OpenAI
import requests
header = {
    "Authorization": Bearer $OPENAI_API_KEY"
}
reqs=requests.get(url="https://api.openai.com/v1/models?access_token=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",verify=False)
print(reqs)