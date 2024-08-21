from flask import Flask
from flask_restful import Resource, Api
from transformers import AutoTokenizer,GPTJForCausalLM
import torch

app: Flask = Flask(__name__)
api: Api = Api(app)
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

class TestRoute(Resource):
    def get(self, input_str: str) -> dict[str, str]:
        return {'Hello': input_str}

class AskGPTJ(Resource):
    def get(self, prompt: str) -> str:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(input_ids)
        return tokenizer.batch_decode(gen_tokens)[0]

api.add_resource(TestRoute, '/<string:input_str>')
api.add_resource(AskGPTJ, '/prompt/<string:prompt>')

if __name__ == '__main__':
    app.run(debug=True)
