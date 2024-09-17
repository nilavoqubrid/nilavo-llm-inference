import subprocess
import sys

try:
    import llmverse
    print("llmverse is already installed.")
except ImportError:
    print("llmverse is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llmverse"])
    print("llmverse installed successfully!")
print("Running the rest of your Python script...")



from flask import Flask, request, jsonify
from llmverse import download_hf_model, load_model, get_response

app = Flask(__name__)

@app.route('/inf', methods=['POST'])
def generate_response():
    # global model, tokenizer

    data = request.json
    model_id = data.get("model_id")
    hf_token = data.get("hf_token")
    optimize = data.get("optimize", "4-bit")
    use_flash_attn = data.get("use_flash_attn", False)

    prompt = data.get("prompt", "Hello!")
    # Validate and set default values for optional parameters
    try:
        max_new_tokens = int(data.get("max_new_tokens", 500))
        temperature = float(data.get("temperature", 1.0))
        top_p = float(data.get("top_p", 1.0))
        repetition_penalty = float(data.get("repetition_penalty", 1.0))
        
        # Validate optimize parameter
        valid_optimizes = ["4-bit", "8-bit", "16-bit", None]
        if optimize not in valid_optimizes:
            raise ValueError(f"Invalid optimize value. Allowed values are {valid_optimizes}.")

         # Validate use_flash_attn parameter
        if not isinstance(use_flash_attn, bool):
            raise ValueError("use_flash_attn must be a boolean.")

        # Check ranges
        if max_new_tokens < 1:
            raise ValueError("Max_new_tokens must be a positive number.")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0.")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("Top_p must be between 0.0 and 1.0.")
        if not (0.0 <= repetition_penalty <= 2.0):
            raise ValueError("Repetition_penalty must be between 0.0 and 2.0.")


        local_dir = model_id.split('/')[-1]
        # Download and load the model
        download_hf_model(model_id=model_id, local_dir=local_dir, hf_token=hf_token)
        model, tokenizer = load_model(model_path=local_dir, optimize=optimize, device="auto", use_flash_attn=use_flash_attn)

        # Generate response using the model
        response = get_response(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

        
    return jsonify({"response": response}), 200

if __name__ == '__main__':
    # Run the app on all available IP addresses on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
