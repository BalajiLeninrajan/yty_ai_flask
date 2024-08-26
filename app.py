from flask import Flask, request, jsonify
from transformers import AutoTokenizer, GPTJForCausalLM, AutoModelForCausalLM
import torch

app: Flask = Flask(__name__)

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token

fine_tuned_model = AutoModelForCausalLM.from_pretrained("./ai_model/gpt-j-finetuned")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./ai_model/gpt-j-finetuned")
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Route not found"}), 404


@app.route('/', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        inputs = fine_tuned_tokenizer(prompt, return_tensors="pt")
        output = fine_tuned_model.generate(**inputs, max_length=100)
        result = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/noFineTune', methods=['POST'])
def generateNoFineTune():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_length=100)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8086, debug=True)
